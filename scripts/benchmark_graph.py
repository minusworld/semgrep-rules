import base64
import io
import json
import os
import sys
import tempfile
import requests
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from matplotlib import rcParams
from pathlib import Path
from functools import lru_cache

from typing import Dict, Any, Optional, Tuple

from scripts.util import ArchList

plt.style.use("ggplot")
rcParams.update({'figure.autolayout': True})

def _read_config(config_path):
    with open(config_path, 'r') as fin:
        return yaml.safe_load(fin)

EXTENSION_MAP = {'python': 'py', 'javascript': 'js', 'go': 'go', 'c': 'c', 'java': 'java', 'ruby': 'ruby'}

@lru_cache(32)
def get_sloc(repo_target: Tuple[str, str], language: str) -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_url = repo_target[0]
        commit_hash = repo_target[1]
        name = repo_url.split('/')[-1]
        target_dir = os.path.join(temp_dir, name)
        subprocess.run(
            ["git", "clone", repo_url, target_dir]
        )
        if commit_hash:
            subprocess.run(
                ["git", "checkout", commit_hash],
                cwd=target_dir
            )
        subprocess.run(
            ["git", "clean", "-xdf"],
            cwd=target_dir
        )

        sloc_results = subprocess.check_output([
            "sloc",
            "--format=json",
            target_dir
        ])
        sloc_results = json.loads(sloc_results)
        # Switch these to get a more accurate X-axis measurement if you care.
        # Otherwise, total sloc is fine for comparing things to each other
        #sloc = sloc_results.get('byExt', {}).get(EXTENSION_MAP.get(language), {}).get('summary', {}).get('source', 0)
        sloc = sloc_results.get('summary', {}).get('source', 0)
        return sloc

def generate_plot(df: pd.DataFrame) -> io.BytesIO:
    stream = io.BytesIO()
    ax = df.plot.barh(
        x="benchmark",
        y="sec per sloc",
        figsize=(16,64),
        title="Semgrep Rules Benchmarks"
    )
    fig = ax.get_figure()
    fig.savefig(stream, format="png")
    stream.seek(0)
    return stream

def get_language(rule_path: Path) -> Optional[str]:
    with open(rule_path, 'r') as fin:
        rules = yaml.safe_load(fin)
    return ArchList(rules.get('rules', [])).get(0, {}).get('languages', [None])[0]

#def tablify_stats_json(data: str) ->  bytes:
#    data = json.loads(data)
#    benchmarks = data.get('benchmarks')
#    benchmarks = list(filter(lambda b: "public_repo" in b.get('fullname', ""), benchmarks))
#    table = []
#    for benchmark in benchmarks:
#        name = benchmark.get('name')
#        language = get_language(config.keys(), name)
#        repo_target = (
#            ArchList(config.get(language, [])).get(0, {}).get('url'),
#            ArchList(config.get(language, [])).get(0, {}).get('commit')
#        )
#        sloc = get_sloc(repo_target, language)
#        table.append(
#            (name, language, sloc / benchmark.get('stats').get('mean'))
#        )
#    df = pd.DataFrame(table, columns=["benchmark", "language", "sloc/sec"])
#    return generate_plot(df)

def tablify_stats_text(config: Dict[str, str], text: str) ->  bytes:
    import re
    table = []
    for line in text.split('\n'):
        try:
            cols = re.split("\s+", line)
            test_name = cols[0]
            rule_path = test_name[ test_name.find('[') + 1 : test_name.find(']') ]
            language = get_language(rule_path)

            running_time = re.match("\((.*?)\)", cols[2]).group(1)
            if running_time.startswith('>'):
                running_time = 1000.0
            else:
                running_time = float(running_time)

            repo_target = (
                ArchList(config.get(language, [])).get(0, {}).get('url'),
                ArchList(config.get(language, [])).get(0, {}).get('commit'),
            )
            sloc = get_sloc(repo_target, language)
        except Exception as e:
            print(repr(e), line)
            continue
        table.append(
            (rule_path, language, running_time / sloc)
        )
    df = pd.DataFrame(table, columns=["benchmark", "language", "sec per sloc"])
    df = df.sort_values(by="sec per sloc")
    return generate_plot(df)

def img_tag(data: bytes) -> str:
    data = base64.b64encode(data)
    return '<img src="data:image/png;base64, {}">'.format(data.decode('utf-8'))

def save(stuff: bytes, filename: str):
    with open(filename, 'wb') as fout:
        fout.write(stuff)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Add arguments here

    parser.add_argument("uri")
    parser.add_argument("config")
    parser.add_argument("--save")
    parser.add_argument("--img-tag", action="store_true")

    args = parser.parse_args()

    config = _read_config(args.config)

    uri = args.uri
    if os.path.exists(uri):
        with open(uri, 'r') as fin:
            data = fin.read()
    else:
        r = requests.get(uri)
        data = r.text

    try:
        data = json.loads(data)
        tablify_stats = tablify_stats_json
    except Exception:
        tablify_stats = tablify_stats_text

    dump = tablify_stats(config, data).read()

    if args.img_tag:
        dump = img_tag(dump)
    
    if args.save:
        save(dump, args.save)
    else:
        print(dump)
