#!/usr/bin/env python3
"""
Manage Grafana alert rules and ntfy contact point from config.yaml.

Usage:
  manage-alerts.py --create [-c config.yaml] [--test]   # create/update alerts
  manage-alerts.py --delete [-c config.yaml]            # delete alerts

Reads alert definitions from config.yaml and manages:
  1. A webhook contact point targeting your ntfy server
  2. A notification policy routing to that contact point
  3. Alert rules per the 'alerts' section

Required env vars (loaded from ../.env or exported):
  GRAFANA_URL            - e.g. https://grafana.leh.example.com
  GRAFANA_API_KEY        - service-account token with alerting permissions
  INFLUXDB_DATASOURCE_UID - UID of your InfluxDB datasource in Grafana
  NTFY_URL               - e.g. https://ntfy.leh.example.com
  NTFY_TOPIC             - e.g. trading-signals
"""

import argparse
import json
import os
import sys
from pathlib import Path
from string import Template

import requests
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "alerts" / "config.yaml"
ENV_FILE = SCRIPT_DIR.parent / ".env"

def load_env():
    """Load .env file if it exists (simple KEY=VALUE parser)."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                os.environ.setdefault(key, value)


def resolve_vars(value):
    """Substitute ${VAR} references with env var values."""
    if not isinstance(value, str):
        return value
    # Use shell-style substitution
    result = value
    import re
    for match in re.finditer(r"\$\{(\w+)\}", value):
        var_name = match.group(1)
        env_val = os.environ.get(var_name, "")
        if not env_val:
            print(f"  WARN: ${{{var_name}}} not set in environment")
        result = result.replace(match.group(0), env_val)
    return result


def load_config(config_path: Path):
    """Load and return the YAML config with env var substitution."""
    raw = config_path.read_text()
    cfg = yaml.safe_load(raw)

    # Resolve env vars in grafana and notification sections
    cfg["grafana"]["folder"] = resolve_vars(cfg["grafana"].get("folder", "Stock Alerts"))
    cfg["grafana"]["datasource_uid"] = resolve_vars(cfg["grafana"]["datasource_uid"])

    ntfy = cfg["notification"]["ntfy"]
    ntfy["url"] = resolve_vars(ntfy["url"])
    ntfy["topic"] = resolve_vars(ntfy["topic"])

    return cfg


class GrafanaClient:
    def __init__(self, url: str, api_key: str):
        self.base_url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Disable-Provenance": "true",  # allows UI editing later
        })

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        resp = self.session.request(method, self._url(path), **kwargs)
        return resp

    # --- Folders ---
    def get_or_create_folder(self, title: str) -> str:
        """Get folder UID by title, or create it. Returns UID."""
        resp = self._request("GET", "/api/search", params={"type": "dash-folder", "query": title})
        if resp.status_code == 200:
            for folder in resp.json():
                if folder.get("title") == title:
                    return folder["uid"]

        # Create folder
        payload = {"title": title}
        resp = self._request("POST", "/api/folders", json=payload)
        if resp.status_code in (200, 412):
            # 412 = already exists (race condition)
            if resp.status_code == 412:
                return self.get_or_create_folder(title)
            return resp.json()["uid"]
        resp.raise_for_status()

    # --- Contact Points ---
    def list_contact_points(self):
        resp = self._request("GET", "/api/v1/provisioning/contact-points")
        resp.raise_for_status()
        return resp.json()

    def upsert_contact_point(self, name: str, settings: dict) -> str:
        """Create or update a webhook contact point. Returns UID."""
        existing = self.list_contact_points()
        for cp in existing:
            if cp.get("name") == name:
                uid = cp["uid"]
                payload = {
                    "name": name,
                    "type": "webhook",
                    "settings": settings,
                    "uid": uid,
                }
                resp = self._request("PUT", f"/api/v1/provisioning/contact-points/{uid}", json=payload)
                if resp.status_code == 202:
                    print(f"  Updated contact point: {name} (uid={uid})")
                    return uid
                resp.raise_for_status()

        # Create new
        payload = {
            "name": name,
            "type": "webhook",
            "settings": settings,
        }
        resp = self._request("POST", "/api/v1/provisioning/contact-points", json=payload)
        if resp.status_code in (201, 202):
            uid = resp.json().get("uid", "unknown")
            print(f"  Created contact point: {name} (uid={uid})")
            return uid
        resp.raise_for_status()

    # --- Notification Policies ---
    def set_notification_policy(self, contact_point: str, match_labels: dict = None):
        """Add a child policy routing matching alerts to the contact point."""
        resp = self._request("GET", "/api/v1/provisioning/policies")
        resp.raise_for_status()
        policy_tree = resp.json()

        # Check if route already exists
        routes = policy_tree.get("routes") or []
        for route in routes:
            if route.get("receiver") == contact_point:
                print(f"  Notification policy for '{contact_point}' already exists.")
                return

        # Add new child route
        new_route = {
            "receiver": contact_point,
            "matchers": [f"{k}={v}" for k, v in (match_labels or {}).items()],
            "continue": False,
        }
        if not match_labels:
            new_route["matchers"] = ["category=signals"]

        routes.append(new_route)
        policy_tree["routes"] = routes

        resp = self._request("PUT", "/api/v1/provisioning/policies", json=policy_tree)
        if resp.status_code in (200, 202):
            print(f"  Notification policy updated — routing 'category=signals' → {contact_point}")
        else:
            print(f"  WARN: Failed to update notification policy: {resp.status_code}")
            print(f"  {resp.text}")

    # --- Delete ---
    def delete_contact_point(self, name: str) -> bool:
        """Delete a contact point by name. Returns True if deleted."""
        existing = self.list_contact_points()
        for cp in existing:
            if cp.get("name") == name:
                uid = cp["uid"]
                resp = self._request("DELETE", f"/api/v1/provisioning/contact-points/{uid}")
                if resp.status_code in (200, 202, 204):
                    print(f"  Deleted contact point: {name} (uid={uid})")
                    return True
                else:
                    print(f"  WARN: Failed to delete contact point '{name}': HTTP {resp.status_code}")
                    return False
        print(f"  Contact point '{name}' not found — nothing to delete.")
        return True

    def remove_notification_policy(self, contact_point: str):
        """Remove the child policy routing to the given contact point."""
        resp = self._request("GET", "/api/v1/provisioning/policies")
        resp.raise_for_status()
        policy_tree = resp.json()

        routes = policy_tree.get("routes") or []
        new_routes = [r for r in routes if r.get("receiver") != contact_point]

        if len(new_routes) == len(routes):
            print(f"  No notification policy for '{contact_point}' found — nothing to remove.")
            return

        policy_tree["routes"] = new_routes
        resp = self._request("PUT", "/api/v1/provisioning/policies", json=policy_tree)
        if resp.status_code in (200, 202):
            print(f"  Removed notification policy for '{contact_point}'.")
        else:
            print(f"  WARN: Failed to update notification policy: {resp.status_code}")

    def delete_alert_rule(self, rule_name: str) -> bool:
        """Delete an alert rule by title. Returns True if deleted."""
        existing = self.list_alert_rules()
        for rule in existing:
            if rule.get("title") == rule_name:
                uid = rule["uid"]
                resp = self._request("DELETE", f"/api/v1/provisioning/alert-rules/{uid}")
                if resp.status_code in (200, 202, 204):
                    print(f"  Deleted alert rule: {rule_name} (uid={uid})")
                    return True
                else:
                    print(f"  WARN: Failed to delete alert rule '{rule_name}': HTTP {resp.status_code}")
                    return False
        print(f"  Alert rule '{rule_name}' not found — nothing to delete.")
        return True

    # --- Test ---
    def test_contact_point(self, receiver_name: str, webhook_settings: dict, labels: dict = None, annotations: dict = None):
        """Send a test notification through the Grafana alertmanager receiver test endpoint."""
        payload = {
            "receivers": [{
                "name": receiver_name,
                "grafana_managed_receiver_configs": [{
                    "name": receiver_name,
                    "type": "webhook",
                    "settings": webhook_settings,
                }]
            }],
            "alert": {
                "annotations": annotations or {"summary": "TEST: Buy Signal for AAPL"},
                "labels": labels or {
                    "alertname": "Buy Signal",
                    "ticker": "AAPL",
                    "severity": "critical",
                    "category": "signals",
                },
            },
        }
        resp = self._request("POST", "/api/alertmanager/grafana/config/api/v1/receivers/test", json=payload)
        if resp.status_code == 200:
            print("  OK — test notification sent successfully.")
            return True
        elif resp.status_code == 207:
            # Multi-status: some receivers may have failed
            results = resp.json()
            print(f"  PARTIAL — {resp.status_code}: {json.dumps(results, indent=2)[:300]}")
            return True
        else:
            print(f"  FAIL — HTTP {resp.status_code}: {resp.text[:300]}")
            return False

    # --- Alert Rules ---
    def list_alert_rules(self):
        resp = self._request("GET", "/api/v1/provisioning/alert-rules")
        resp.raise_for_status()
        return resp.json()

    def upsert_alert_rule(self, rule_name: str, folder_uid: str, datasource_uid: str,
                          query: str, alert_cfg: dict):
        """Create or update an alert rule."""
        existing = self.list_alert_rules()
        existing_uid = None
        for rule in existing:
            if rule.get("title") == rule_name:
                existing_uid = rule["uid"]
                break

        eval_interval = alert_cfg.get("evaluation_interval", "5m")
        pending_for = alert_cfg.get("for", "0s")
        annotations = alert_cfg.get("annotations", {})
        labels = alert_cfg.get("labels", {})
        labels["severity"] = alert_cfg.get("severity", "warning")

        condition_cfg = alert_cfg.get("condition", {})

        # Support 'above' (gt) and 'below' (lt) thresholds
        # 'below' is useful for alerts like earnings_approaching where
        # days_threshold controls when it fires
        if "below" in condition_cfg:
            days_threshold = alert_cfg.get("days_threshold", condition_cfg["below"])
            evaluator_type = "lt"
            threshold = days_threshold
        else:
            evaluator_type = "gt"
            threshold = condition_cfg.get("above", 0)

        # Build the rule payload (Grafana unified alerting format)
        rule_payload = {
            "title": rule_name,
            "folderUID": folder_uid,
            "ruleGroup": "trading-signals",
            "condition": "C",
            "noDataState": "OK",
            "execErrState": "OK",
            "for": pending_for,
            "annotations": annotations,
            "labels": labels,
            "data": [
                {
                    "refId": "A",
                    "datasourceUid": datasource_uid,
                    "queryType": "",
                    "model": {
                        "query": query,
                        "rawQuery": True,
                        "intervalMs": 1000,
                        "maxDataPoints": 43200,
                        "refId": "A",
                    },
                    "relativeTimeRange": {
                        "from": 86400,  # 1 day in seconds
                        "to": 0,
                    },
                },
                {
                    "refId": "B",
                    "datasourceUid": "__expr__",
                    "queryType": "",
                    "model": {
                        "type": "reduce",
                        "expression": "A",
                        "reducer": "last",
                        "settings": {
                            "mode": "dropNN",
                        },
                        "refId": "B",
                    },
                    "relativeTimeRange": {
                        "from": 0,
                        "to": 0,
                    },
                },
                {
                    "refId": "C",
                    "datasourceUid": "__expr__",
                    "queryType": "",
                    "model": {
                        "type": "threshold",
                        "expression": "B",
                        "conditions": [
                            {
                                "evaluator": {
                                    "type": evaluator_type,
                                    "params": [threshold],
                                },
                                "operator": {"type": "and"},
                                "reducer": {"type": "last"},
                            }
                        ],
                        "refId": "C",
                    },
                    "relativeTimeRange": {
                        "from": 0,
                        "to": 0,
                    },
                },
            ],
        }

        if existing_uid:
            resp = self._request("PUT", f"/api/v1/provisioning/alert-rules/{existing_uid}",
                                 json=rule_payload)
            if resp.status_code in (200, 201, 202):
                print(f"  Updated alert rule: {rule_name} (uid={existing_uid})")
                return
        else:
            resp = self._request("POST", "/api/v1/provisioning/alert-rules", json=rule_payload)
            if resp.status_code in (200, 201, 202):
                uid = resp.json().get("uid", "unknown")
                print(f"  Created alert rule: {rule_name} (uid={uid})")
                return

        print(f"  ERROR: Failed to upsert alert rule '{rule_name}': HTTP {resp.status_code}")
        print(f"  {resp.text[:500]}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manage Grafana alert rules and ntfy contact point from a YAML config."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--create",
        action="store_true",
        help="Create or update alerts, contact point, and notification policy.",
    )
    action.add_argument(
        "--delete",
        action="store_true",
        help="Delete alerts, contact point, and notification policy.",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to the alert config YAML file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        default=False,
        help="After creating, send a test notification through the contact point.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.resolve()

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    load_env()

    # Validate required env vars
    required = ["GRAFANA_URL", "GRAFANA_API_KEY", "INFLUXDB_DATASOURCE_UID", "NTFY_URL", "NTFY_TOPIC"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print("ERROR: Missing required environment variables:")
        for v in missing:
            print(f"  {v}")
        print(f"\nExport them or add them to {ENV_FILE}")
        sys.exit(1)

    cfg = load_config(config_path)
    print(f"Config: {config_path}")
    grafana_url = os.environ["GRAFANA_URL"]
    grafana_key = os.environ["GRAFANA_API_KEY"]

    client = GrafanaClient(grafana_url, grafana_key)
    contact_point_name = cfg["notification"]["contact_point"]

    if args.create:
        _create_alerts(client, cfg, contact_point_name, args)
    elif args.delete:
        _delete_alerts(client, cfg, contact_point_name)


def _create_alerts(client, cfg, contact_point_name, args):
    """Create or update alerts, contact point, and notification policy."""
    print(f"=== Creating alerts on {client.base_url} ===\n")

    # 1. Create/update ntfy contact point
    print("[1/3] Contact Point (ntfy webhook)")
    ntfy_cfg = cfg["notification"]["ntfy"]
    ntfy_topic = ntfy_cfg["topic"]
    ntfy_url = ntfy_cfg["url"]
    ntfy_priority = ntfy_cfg.get("priority", 4)
    ntfy_tags = ntfy_cfg.get("tags", ["warning"])

    # ntfy JSON publish API: POST a JSON body to the base URL.
    # The `message` field in Grafana webhook settings is a Go template
    # whose evaluated output becomes the HTTP body sent to the URL.
    # We construct a Go template that produces valid ntfy JSON.
    ntfy_title = (
        '{{ if eq .Status "firing" }}'
        '{{ .CommonAnnotations.summary }}'
        '{{ else }}'
        '[OK] {{ .CommonLabels.alertname }}'
        '{{ end }}'
    )
    ntfy_body = (
        '{{ if eq .Status "firing" }}'
        '{{ range .Alerts }}'
        '**{{ .Labels.alertname }}** - {{ .Labels.ticker }}\\n\\n'
        '{{ .Annotations.description }}\\n\\n'
        '_Severity: {{ .Labels.severity }}_'
        '{{ end }}'
        '{{ else }}'
        '**RESOLVED**: {{ .CommonLabels.alertname }}'
        '{{ end }}'
    )
    ntfy_click = '{{ .ExternalURL }}'

    # Build the Go template that outputs ntfy JSON
    ntfy_message_tpl = (
        '{'
        f'"topic":"{ntfy_topic}",'
        f'"title":"{ntfy_title}",'
        f'"message":"{ntfy_body}",'
        f'"tags":{json.dumps(ntfy_tags)},'
        f'"priority":{ntfy_priority},'
        '"markdown":true,'
        f'"click":"{ntfy_click}"'
        '}'
    )

    webhook_settings = {
        "url": f"{ntfy_url}",
        "httpMethod": "POST",
        "maxAlerts": "1",
        "message": ntfy_message_tpl,
    }
    client.upsert_contact_point(contact_point_name, webhook_settings)
    print()

    # 2. Set up notification policy
    print("[2/3] Notification Policy")
    client.set_notification_policy(
        contact_point_name,
        match_labels={"category": "signals"},
    )
    print()

    # 3. Create/update alert rules
    print("[3/3] Alert Rules")
    folder_title = cfg["grafana"]["folder"]
    datasource_uid = cfg["grafana"]["datasource_uid"]
    folder_uid = client.get_or_create_folder(folder_title)
    print(f"  Folder: {folder_title} (uid={folder_uid})")

    alerts = cfg.get("alerts", {})
    errors = 0
    for name, alert_cfg in alerts.items():
        if not alert_cfg.get("enabled", True):
            print(f"  Skipping disabled alert: {name}")
            continue

        rule_name = name.replace("_", " ").title()
        query = alert_cfg.get("query", "")
        if not query:
            print(f"  WARN: No query for alert '{name}', skipping.")
            continue

        result = client.upsert_alert_rule(rule_name, folder_uid, datasource_uid, query, alert_cfg)
        if result is False:
            errors += 1

    print()
    if errors:
        print(f"Done with {errors} error(s).")
        sys.exit(1)
    else:
        print("Done — all alerts created.")

    # 4. Test notification (if --test)
    if args.test:
        print()
        print("[TEST] Sending test notification through contact point...")
        test_labels = {
            "alertname": "Buy Signal",
            "ticker": "TEST",
            "severity": "critical",
            "category": "signals",
            "team": "trading",
        }
        test_annotations = {
            "summary": "TEST: BUY SIGNAL for TEST ticker",
            "description": "This is a test notification from manage-alerts.py --test",
        }
        success = client.test_contact_point(
            contact_point_name, webhook_settings,
            labels=test_labels, annotations=test_annotations,
        )
        if not success:
            sys.exit(1)
        print("  Check your ntfy client for the test notification.")


def _delete_alerts(client, cfg, contact_point_name):
    """Delete alerts, contact point, and notification policy."""
    print(f"=== Deleting alerts on {client.base_url} ===\n")

    # 1. Delete alert rules
    alerts = cfg.get("alerts", {})
    rule_names = [name.replace("_", " ").title() for name in alerts]
    print(f"[1/3] Alert Rules ({len(rule_names)} defined in config)")
    errors = 0
    for rule_name in rule_names:
        if not client.delete_alert_rule(rule_name):
            errors += 1
    print()

    # 2. Remove notification policy
    print("[2/3] Notification Policy")
    client.remove_notification_policy(contact_point_name)
    print()

    # 3. Delete contact point
    print("[3/3] Contact Point")
    client.delete_contact_point(contact_point_name)
    print()

    if errors:
        print(f"Done with {errors} error(s).")
        sys.exit(1)
    else:
        print("Done — all alerts deleted.")


if __name__ == "__main__":
    main()
