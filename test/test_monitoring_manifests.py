"""Static checks that the monitoring stack is wired consistently with the API."""
import json
import re
from pathlib import Path

import pytest
import yaml
from prometheus_client import REGISTRY, generate_latest

import src.api.observability  # noqa: F401  (registers the API metrics)

K8S = Path("k8s")
MON = K8S / "monitoring"


def _yaml(path: Path):
    return yaml.safe_load(path.read_text())


def _yaml_all(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def _by_kind(docs: list[dict], kind: str) -> dict:
    return next(d for d in docs if d["kind"] == kind)


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads((MON / "grafana" / "dashboards" / "heart-api.json").read_text())


@pytest.fixture(scope="module")
def scrape() -> dict:
    return _yaml(MON / "prometheus" / "prometheus.yml")


def test_kustomization_lists_only_existing_files():
    kust = _yaml(MON / "kustomization.yaml")
    for name in kust["resources"]:
        assert (MON / name).exists(), name
    for gen in kust["configMapGenerator"]:
        for name in gen["files"]:
            assert (MON / name).exists(), name


def test_prometheus_discovers_the_api_service_and_port(scrape):
    service = _yaml(K8S / "service.yaml")
    job = next(j for j in scrape["scrape_configs"] if j["job_name"] == "heart-api")
    assert job["metrics_path"] == "/metrics"
    assert job["kubernetes_sd_configs"][0]["namespaces"]["names"] == [
        service["metadata"]["namespace"]]
    keep = {r["source_labels"][0]: r["regex"] for r in job["relabel_configs"]
            if r.get("action") == "keep"}
    assert keep["__meta_kubernetes_service_name"] == service["metadata"]["name"]
    assert keep["__meta_kubernetes_endpoint_port_name"] == service["spec"]["ports"][0]["name"]


def test_prometheus_rbac_covers_the_scraped_namespace_and_service_account():
    docs = _yaml_all(MON / "prometheus-discovery.yaml")
    role, binding = _by_kind(docs, "Role"), _by_kind(docs, "RoleBinding")
    deploy_ns = _yaml(K8S / "deployment.yaml")["metadata"]["namespace"]
    prom_docs = _yaml_all(MON / "prometheus.yaml")
    sa = _by_kind(prom_docs, "ServiceAccount")["metadata"]
    assert role["metadata"]["namespace"] == deploy_ns
    assert "endpoints" in role["rules"][0]["resources"]
    assert set(role["rules"][0]["verbs"]) <= {"get", "list", "watch"}  # read-only
    assert binding["subjects"][0]["name"] == sa["name"]
    assert binding["subjects"][0]["namespace"] == sa["namespace"]
    assert binding["roleRef"]["name"] == role["metadata"]["name"]


def test_prometheus_deployment_uses_the_mounted_config_and_service_account():
    docs = _yaml_all(MON / "prometheus.yaml")
    spec = _by_kind(docs, "Deployment")["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert spec["serviceAccountName"] == _by_kind(docs, "ServiceAccount")["metadata"]["name"]
    mount = next(m for m in container["volumeMounts"] if m["name"] == "config")
    assert f"--config.file={mount['mountPath']}/prometheus.yml" in container["args"]
    assert container["readinessProbe"]["httpGet"]["path"] == "/-/ready"


def test_grafana_datasource_points_at_the_prometheus_service():
    docs = _yaml_all(MON / "prometheus.yaml")
    svc = _by_kind(docs, "Service")
    port = svc["spec"]["ports"][0]["port"]
    ds = _yaml(MON / "grafana" / "datasource.yaml")["datasources"][0]
    expected = f"http://{svc['metadata']['name']}.{svc['metadata']['namespace']}.svc:{port}"
    assert ds["url"] == expected
    assert ds["type"] == "prometheus" and ds["isDefault"] is True


def test_dashboard_uses_the_provisioned_datasource_uid(dashboard):
    uid = _yaml(MON / "grafana" / "datasource.yaml")["datasources"][0]["uid"]
    for p in dashboard["panels"]:
        assert p["datasource"]["uid"] == uid
        for t in p["targets"]:
            assert t["datasource"]["uid"] == uid


def test_grafana_serves_dashboards_from_the_provider_path():
    provider = _yaml(MON / "grafana" / "dashboards.yaml")["providers"][0]["options"]["path"]
    deploy = _by_kind(_yaml_all(MON / "grafana.yaml"), "Deployment")
    container = deploy["spec"]["template"]["spec"]["containers"][0]
    assert any(m["mountPath"] == provider for m in container["volumeMounts"])
    home = next(e["value"] for e in container["env"]
                if e["name"] == "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH")
    assert home == f"{provider}/heart-api.json"


def test_dashboard_panels_are_well_formed(dashboard):
    ids = [p["id"] for p in dashboard["panels"]]
    assert len(ids) == len(set(ids))
    assert dashboard["uid"] == "heart-api"
    for p in dashboard["panels"]:
        assert p["title"] and p["targets"]
        x, w = p["gridPos"]["x"], p["gridPos"]["w"]
        assert 0 <= x and x + w <= 24


def test_every_dashboard_query_uses_metrics_the_api_actually_exports(dashboard):
    exposed = set(re.findall(r"^(?:# TYPE )?([a-z_:]+)[ {]",
                             generate_latest(REGISTRY).decode(), re.M))
    exposed |= {"up", "process_cpu_seconds_total", "process_resident_memory_bytes"}
    used = set()
    for p in dashboard["panels"]:
        for t in p["targets"]:
            used |= set(re.findall(r"\b((?:api|process)_[a-z_]+|up)\b", t["expr"]))
    assert used, "dashboard has no queries"
    missing = sorted(m for m in used if m not in exposed)
    assert not missing, f"dashboard queries metrics the API does not export: {missing}"


def test_ingress_hosts_route_to_existing_services():
    ingress = _yaml(MON / "ingress.yaml")
    services = {
        "grafana": _by_kind(_yaml_all(MON / "grafana.yaml"), "Service"),
        "prometheus": _by_kind(_yaml_all(MON / "prometheus.yaml"), "Service"),
    }
    hosts = {}
    for rule in ingress["spec"]["rules"]:
        backend = rule["http"]["paths"][0]["backend"]["service"]
        hosts[rule["host"]] = backend
        svc = services[backend["name"]]
        assert backend["port"]["name"] in {p["name"] for p in svc["spec"]["ports"]}
        assert svc["metadata"]["namespace"] == ingress["metadata"]["namespace"]
    assert set(hosts) == {"grafana.localhost", "prometheus.localhost"}


def test_monitoring_scripts_use_the_same_hosts_and_credentials():
    script = Path("scripts/k8s_monitoring.sh").read_text()
    assert "grafana.localhost" in script and "prometheus.localhost" in script
    env = _by_kind(_yaml_all(MON / "grafana.yaml"), "Deployment")[
        "spec"]["template"]["spec"]["containers"][0]["env"]
    password = next(e["value"] for e in env if e["name"] == "GF_SECURITY_ADMIN_PASSWORD")
    assert f'GRAFANA_PASSWORD="{password}"' in script


def test_api_pods_log_json_in_the_cluster():
    env = _yaml(K8S / "deployment.yaml")["spec"]["template"]["spec"]["containers"][0]["env"]
    assert {"name": "LOG_FORMAT", "value": "json"} in env
