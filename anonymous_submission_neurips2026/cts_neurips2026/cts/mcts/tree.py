"""Minimal MCTS tree (expand / attach children)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from cts.types import TreeNode


@dataclass
class SearchTree:
    nodes: List[TreeNode] = field(default_factory=list)

    def new_node(self, text: str, z_star, depth: int, parent_id: Optional[int], W: int = 3) -> int:
        nid = len(self.nodes)
        self.nodes.append(
            TreeNode(
                text_state=text,
                z_star=z_star,
                depth=depth,
                parent_id=parent_id,
                node_id=nid,
                mcts_W=W,
            )
        )
        if parent_id is not None:
            self.nodes[parent_id].children_ids.append(nid)
        return nid

    def root(self) -> TreeNode:
        return self.nodes[0]
