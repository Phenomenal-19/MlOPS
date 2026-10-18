"""Static checks that the Kubernetes manifests agree with each other and the API."""
from pathlib import Path

import pytest
import yaml

K8S = Path("k8s")


def _load(name: str) -> dict:
    return yaml.safe_load((K8S / name).read_text())


@pytest.fixture(scope="module")
def deployment():
    return _load("deployment.yaml")


@pytest.fixture(scope="module")
def service():
    return _load("service.yaml")


@pytest.fixture(scope="module")
def ingress():
    return _load("ingress.yaml")


def _container(deployment: dict) -> dict:
    return deployment["spec"]["template"]["spec"]["containers"][0]


def test_every_manifest_parses_and_targets_the_same_namespace():
    namespace = _load("namespace.yaml")["metadata"]["name"]
    for name in ("deployment.yaml", "service.yaml", "ingress.yaml"):
        doc = _load(name)
        assert doc["apiVersion"] and doc["kind"]
        assert doc["metadata"]["namespace"] == namespace


def test_deployment_selector_matches_pod_labels(deployment):
    selector = deployment["spec"]["selector"]["matchLabels"]
    labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert selector.items() <= labels.items()


def test_service_selects_the_deployment_pods(deployment, service):
    labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert service["spec"]["selector"].items() <= labels.items()


def test_service_targets_the_named_container_port(deployment, service):
    port_names = {p["name"] for p in _container(deployment)["ports"]}
    assert {p["targetPort"] for p in service["spec"]["ports"]} <= port_names


def test_ingress_routes_to_the_service_and_its_port(service, ingress):
    backend = ingress["spec"]["rules"][0]["http"]["paths"][0]["backend"]["service"]
    assert backend["name"] == service["metadata"]["name"]
    assert backend["port"]["name"] in {p["name"] for p in service["spec"]["ports"]}


def test_probes_hit_the_health_endpoint(deployment):
    c = _container(deployment)
    assert c["readinessProbe"]["httpGet"]["path"] == "/health"
    assert c["livenessProbe"]["httpGet"]["path"] == "/health"


def test_container_port_matches_dockerfile_expose(deployment):
    expose = [ln for ln in Path("Dockerfile").read_text().splitlines()
              if ln.startswith("EXPOSE")][0].split()[1]
    assert str(_container(deployment)["ports"][0]["containerPort"]) == expose


def test_runs_replicated_and_without_downtime_on_rollout(deployment):
    assert deployment["spec"]["replicas"] >= 2
    assert deployment["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0


def test_resources_are_bounded_and_user_is_non_root(deployment):
    c = _container(deployment)
    assert {"cpu", "memory"} <= set(c["resources"]["requests"])
    assert {"cpu", "memory"} <= set(c["resources"]["limits"])
    pod_sec = deployment["spec"]["template"]["spec"]["securityContext"]
    assert pod_sec["runAsNonRoot"] is True
    assert pod_sec["runAsUser"] == 10001  # matches useradd --uid in the Dockerfile
    assert "useradd --create-home --uid 10001" in Path("Dockerfile").read_text()


def test_local_image_is_not_force_pulled(deployment):
    # The image is loaded into kind, so the default Always policy would fail.
    assert _container(deployment)["imagePullPolicy"] in {"IfNotPresent", "Never"}


def test_kind_host_port_matches_the_deploy_script():
    cluster = yaml.safe_load((K8S / "kind" / "cluster.yaml").read_text())
    mapping = cluster["nodes"][0]["extraPortMappings"][0]
    assert f"localhost:{mapping['hostPort']}" in Path("README.md").read_text() \
        if Path("README.md").exists() else True
    assert f"127.0.0.1:{mapping['hostPort']}" in Path("scripts/k8s_deploy.sh").read_text()
