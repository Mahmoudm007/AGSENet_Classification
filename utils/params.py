from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch


def parameter_overview(model: torch.nn.Module) -> Dict[str, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "frozen_parameters": int(frozen_params),
    }


def parameter_breakdown(model: torch.nn.Module) -> List[Dict[str, int]]:
    buckets: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for name, param in model.named_parameters():
        block = name.split(".", 1)[0]
        if block not in buckets:
            buckets[block] = {
                "block_name": block,
                "total_parameters": 0,
                "trainable_parameters": 0,
                "frozen_parameters": 0,
            }
        buckets[block]["total_parameters"] += int(param.numel())
        if param.requires_grad:
            buckets[block]["trainable_parameters"] += int(param.numel())
        else:
            buckets[block]["frozen_parameters"] += int(param.numel())

    for row in buckets.values():
        row["frozen_parameters"] = row["total_parameters"] - row["trainable_parameters"]
    return list(buckets.values())


def append_parameter_reports(
    model: torch.nn.Module,
    output_dir: str,
    epoch: int,
) -> Tuple[str, str]:
    overview_row = {"epoch": int(epoch)}
    overview_row.update(parameter_overview(model))
    breakdown_rows = parameter_breakdown(model)
    for row in breakdown_rows:
        row["epoch"] = int(epoch)

    overview_path = Path(output_dir) / "parameter_overview.csv"
    breakdown_path = Path(output_dir) / "parameter_breakdown.csv"

    overview_df = pd.DataFrame([overview_row])
    breakdown_df = pd.DataFrame(breakdown_rows)

    overview_df.to_csv(
        overview_path,
        mode="a",
        index=False,
        header=not overview_path.exists(),
    )
    breakdown_df.to_csv(
        breakdown_path,
        mode="a",
        index=False,
        header=not breakdown_path.exists(),
    )
    return str(overview_path), str(breakdown_path)
