# 放原始 3D 模型的地方（工作流 1）

每個類別一個資料夾，把下載或掃描的模型直接丟進去：

```
sim/assets/raw/
  pet_bottle/      *.obj *.glb *.gltf *.stl *.ply *.off
  aluminum_can/
  paper_carton/
  glass_bottle/
```

- 自動下載：`uv run --group assets python sim/assets/fetch_objaverse.py`（Objaverse LVIS，每類 `--per-class 25` 個）。
  它會寫好 `sizes.csv`（選讓厚度符合類別的長度）、`objaverse_sources.csv`（作者、授權、網址）和
  `review.html`。**一定要有人在 `review.html` 勾選要留的模型**，下載 `keep.txt` 後跑
  `--apply-review keep.txt` 刪掉沒勾的；重跑不會再下載已經看過的模型。
- 這個資料夾裡的模型**不會進 git**（檔案大、授權各不同）。請放在團隊雲端硬碟同步。
- 每個類別可以放一個 `sizes.csv`，寫實物最長邊（公尺，用尺量）：

  ```
  filename,length_m
  coke_can.glb,0.122
  ```

  沒寫的話用 `trash_sorter/config.py` 裡該類別的平均長度。
- 轉檔（不用手動改模型）：

  ```
  scripts\isaac.cmd sim\assets\convert_assets.py            # 全部轉
  scripts\isaac.cmd sim\assets\convert_assets.py --dry-run  # 只檢查不寫檔
  ```

  結果在 `sim/assets/usd/real/<類別>/`，報告在 `sim/assets/usd/real/report.csv`。
  報告裡 `note` 欄有字的要看一下（厚度跟類別差太多、面數太多等）。
- 有任何 `real` 模型後，`run_sorter.py` 和 `datagen.py` 會自動改用它們（`--assets auto`）。
- 記下每個模型的來源與授權（Objaverse / Sketchfab 的 CC 授權通常要求標註作者）。
