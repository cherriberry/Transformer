import json

from run_keyformer_benchmark import build_policy, load_config
from models.keyformer import FullKVPolicy, KeyformerPolicy, RandomKVPolicy, RecentWindowPolicy


def test_json_and_yaml_configs_load() -> None:
    quick_json = load_config("configs/keyformer_quick.json")
    quick_yaml = load_config("configs/keyformer_quick.yaml")
    final = load_config("configs/keyformer_final.json")
    assert quick_json == quick_yaml
    assert quick_json["dataset"] == "Salesforce/wikitext"
    assert quick_json["dataset_config"] == "wikitext-2-raw-v1"
    assert final["decode_tokens"] == 128
    assert final["prompt_lengths"] == [256, 512, 1024]
    assert final["cache_ratios"] == [0.25, 0.5, 0.75, 1.0]
    assert final["seeds"] == [17, 29, 43]


def test_policy_factory_covers_requested_baselines() -> None:
    cfg = json.loads(open("configs/keyformer_quick.json", encoding="utf-8").read())
    assert isinstance(build_policy("full_kv", 5, 2, cfg, 1), FullKVPolicy)
    assert isinstance(build_policy("keyformer", 5, 2, cfg, 1), KeyformerPolicy)
    assert isinstance(build_policy("recent_window", 5, 2, cfg, 1), RecentWindowPolicy)
    assert isinstance(build_policy("random_plus_recent", 5, 2, cfg, 1), RandomKVPolicy)
