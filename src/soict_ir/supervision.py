# Adapted for the standalone SoICT release (2026): local imports and CLI paths revised.
# MOOSE-Star-derived code; see LICENSE and NOTICE.
import argparse, collections, hashlib, json, os
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from .tree import HierarchicalSearchTree
from .prompts import instruction_prompts, BRANCH_PROMPT
DEFAULT_BRANCH_PROMPT = BRANCH_PROMPT
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def normalize_title(title: str) -> str:
    return " ".join((title or "").lower().strip().split())

def source_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def build_ir_prompt(
    research_question: str,
    background_survey: str,
    candidates: List[Dict[str, str]],
    previous_hypothesis: Optional[str],
    prompt_name: str,
) -> Tuple[str, Dict[str, str]]:
    prompts = instruction_prompts(prompt_name)
    pre_hyp_text = previous_hypothesis if previous_hypothesis else "No previous hypothesis."

    if not candidates or len(candidates) < 2:
        raise ValueError(f"Need 2-26 candidates, got {len(candidates) if candidates else 0}")
    if len(candidates) > len(LABELS):
        raise ValueError(f"Too many candidates: {len(candidates)}")

    candidates_list = []
    label_to_title = {}
    for i, candidate in enumerate(candidates):
        label = LABELS[i]
        candidates_list.append(
            f"### Candidate [{label}]\n"
            f"**Title:** {candidate['title']}\n"
            f"**Abstract:** {candidate['abstract']}"
        )
        label_to_title[label] = candidate["title"]

    full_prompt = (
        prompts[0] + research_question +
        prompts[1] + background_survey +
        prompts[2] + pre_hyp_text +
        prompts[3] + "\n\n".join(candidates_list) +
        prompts[4]
    )
    return full_prompt, label_to_title

def iter_samples(input_dir: Optional[str], input_jsonl: Optional[str]) -> Iterable[Tuple[str, Dict]]:
    if input_dir:
        for path in sorted(glob(os.path.join(input_dir, "*.json"))):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            yield data.get("source_id") or source_id_from_path(path), data

    if input_jsonl:
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield data.get("source_id") or f"sample_{idx}", data

def build_title_index(tree: HierarchicalSearchTree) -> Dict[str, str]:
    title_to_paper_id = {}
    for paper_id, paper in tree.papers.items():
        title_to_paper_id[normalize_title(paper.get("title", ""))] = paper_id
    return title_to_paper_id

def index_tree(root: Dict) -> Tuple[Dict[str, List[Dict]], Dict[str, Set[str]]]:
    paper_to_path: Dict[str, List[Dict]] = {}
    node_to_leaf_papers: Dict[str, Set[str]] = {}

    def visit(node: Dict, path: List[Dict]) -> Set[str]:
        current_path = path + [node]
        if node.get("is_leaf"):
            leaves = {node["paper_id"]}
            paper_to_path[node["paper_id"]] = current_path
        else:
            leaves = set()
            for child in node.get("children", []):
                leaves.update(visit(child, current_path))
        node_to_leaf_papers[node["node_id"]] = leaves
        return leaves

    visit(root, [])
    return paper_to_path, node_to_leaf_papers

def resolve_target_ids(data: Dict, title_to_paper_id: Dict[str, str]) -> Tuple[List[str], List[Dict]]:
    target_ids = []
    missing = []
    for insp in data.get("inspiration", []):
        title = insp.get("found_title") or insp.get("supposed_title") or ""
        paper_id = title_to_paper_id.get(normalize_title(title))
        if paper_id and paper_id not in target_ids:
            target_ids.append(paper_id)
        else:
            missing.append({
                "title": title,
                "reason": "not_found_in_tree" if not paper_id else "duplicate",
            })
    return target_ids, missing

def nodes_for_targets(
    target_ids: Sequence[str],
    paper_to_path: Dict[str, List[Dict]],
) -> List[Dict]:
    nodes_by_id = {}
    for paper_id in target_ids:
        for node in paper_to_path.get(paper_id, []):
            if node.get("is_leaf") or len(node.get("children", [])) <= 1:
                continue
            nodes_by_id[node["node_id"]] = node
    return list(nodes_by_id.values())

def positive_child_indices(
    node: Dict,
    target_ids: Set[str],
    node_to_leaf_papers: Dict[str, Set[str]],
) -> List[int]:
    indices = []
    for idx, child in enumerate(node.get("children", [])):
        child_leaves = node_to_leaf_papers.get(child["node_id"], set())
        if child_leaves.intersection(target_ids):
            indices.append(idx)
    return indices

def first_positive_for_target_order(
    node: Dict,
    ordered_target_ids: Sequence[str],
    node_to_leaf_papers: Dict[str, Set[str]],
) -> Optional[int]:
    for target_id in ordered_target_ids:
        for idx, child in enumerate(node.get("children", [])):
            child_leaves = node_to_leaf_papers.get(child["node_id"], set())
            if target_id in child_leaves:
                return idx
    return None

def choose_positive_index(
    strategy: str,
    positives: Sequence[int],
    node: Dict,
    ordered_target_ids: Sequence[str],
    node_to_leaf_papers: Dict[str, Set[str]],
    source_id: str,
    tree_name: str,
) -> int:
    if not positives:
        raise ValueError("No positive branch candidates.")

    if strategy == "first":
        first_idx = first_positive_for_target_order(node, ordered_target_ids, node_to_leaf_papers)
        if first_idx is None:
            raise ValueError("No positive branch found for target order.")
        return first_idx

    if strategy == "hash":
        key = f"{source_id}|{tree_name}|{node['node_id']}"
        value = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return positives[value % len(positives)]

    raise ValueError(f"Unknown label strategy: {strategy}")

def build_assistant_response(label: str) -> str:
    return (
        "<think>\n"
        "I need to route the hierarchical search using the medoid papers as branch representatives. "
        f"Candidate [{label}] is the branch to continue exploring because it keeps the search aligned "
        "with a useful inspiration for this research context.\n"
        "</think>\n\n"
        f"**Selected ID starts:** [{label}] **Selected ID ends**\n\n"
        f"**Selection Reason starts:** Candidate [{label}] is selected as the branch to continue "
        "searching for a useful inspiration that can address the research problem. **Selection Reason ends**"
    )

def make_example(
    tree: HierarchicalSearchTree,
    tree_name: str,
    tree_seed: Optional[int],
    data: Dict,
    source_id: str,
    node: Dict,
    target_ids: Sequence[str],
    node_to_leaf_papers: Dict[str, Set[str]],
    label_strategy: str,
    prompt_name: str,
) -> Dict:
    candidates = tree.get_candidates_at_node(node)
    if len(candidates) < 2:
        raise ValueError(f"Node {node['node_id']} has fewer than 2 candidates.")
    if len(candidates) > len(LABELS):
        raise ValueError(f"Node {node['node_id']} has too many candidates: {len(candidates)}")

    target_set = set(target_ids)
    positives = positive_child_indices(node, target_set, node_to_leaf_papers)
    selected_idx = choose_positive_index(
        strategy=label_strategy,
        positives=positives,
        node=node,
        ordered_target_ids=target_ids,
        node_to_leaf_papers=node_to_leaf_papers,
        source_id=source_id,
        tree_name=tree_name,
    )
    label = LABELS[selected_idx]

    prompt, _ = build_ir_prompt(
        research_question=data["research_question"],
        background_survey=data["background_survey"],
        candidates=[{"title": c.title, "abstract": c.abstract} for c in candidates],
        previous_hypothesis=None,
        prompt_name=prompt_name,
    )

    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": build_assistant_response(label)},
        ],
        "metadata": {
            "task": "tomatostar_branch_routing_ir",
            "source_id": source_id,
            "tree_name": tree_name,
            "tree_seed": tree_seed,
            "node_id": node["node_id"],
            "node_level": node.get("level"),
            "candidate_count": len(candidates),
            "positive_indices": list(positives),
            "positive_labels": [LABELS[i] for i in positives],
            "selected_index": selected_idx,
            "selected_label": label,
            "target_paper_ids": list(target_ids),
        },
    }

def write_dataset_info(output_dir: str, split_name: str, file_name: str) -> None:
    payload = {
        split_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_dataset(input_jsonl, tree_dir, output_dir, max_examples=10000):
    """Expand in source order, then cap decisions (including a partial final record)."""
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    from pathlib import Path
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    tree = HierarchicalSearchTree.load(str(tree_dir))
    titles = build_title_index(tree)
    paths, leaves = index_tree(tree.root)
    output.mkdir(parents=True)
    count = 0
    sources = set()
    depths = collections.Counter()
    missing_counts = collections.Counter()
    with (output / "train.jsonl").open("w", encoding="utf-8") as handle:
        for source_id, data in iter_samples(None, str(input_jsonl)):
            target_ids, missing = resolve_target_ids(data, titles)
            missing_counts.update(item["reason"] for item in missing)
            for node in nodes_for_targets(target_ids, paths):
                if count == max_examples:
                    break
                row = make_example(tree, Path(tree_dir).name, tree.config.get("seed"),
                                   data, source_id, node, target_ids, leaves, "first", BRANCH_PROMPT)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                sources.add(source_id)
                depths[str(node["level"])] += 1
            if count == max_examples:
                break
    write_dataset_info(str(output), "train", "train.jsonl")
    summary = {"num_examples": count, "unique_source_records": len(sources),
               "examples_by_depth": dict(depths), "missing_targets": dict(missing_counts),
               "selection": "source-order expansion then decision prefix",
               "tree_name": Path(tree_dir).name, "label_strategy": "first"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if count != max_examples:
        raise ValueError(f"Requested {max_examples} examples, found {count}; output is incomplete")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build SoICT single-positive child-subtree SFT data.")
    parser.add_argument("--input-jsonl", required=True, help="TOMATO-Star training records only")
    parser.add_argument("--tree-dir", required=True, help="Historical TRAINING tree, see docs/reproduction.md")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.input_jsonl, args.tree_dir, args.output_dir, args.max_examples), indent=2))


if __name__ == "__main__":
    main()
