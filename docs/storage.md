# 大檔案存放（Google Drive）

資料集、模型權重、3D 資產原始檔、影片都不進 git，統一放團隊的 Google Drive 資料夾 `trash-sorter-data`。

## 本機設定（每人一次）

1. 安裝「Google 雲端硬碟電腦版」，登入有 `trash-sorter-data` 編輯權限的帳號。
   - 學校帳號能建「共用雲端硬碟」就用共用雲端硬碟；不行的話由 PM 建一般資料夾分享給大家，各自在網頁上對它按「新增捷徑到雲端硬碟」。
2. 在檔案總管找到 `trash-sorter-data` 的本機路徑（例如 `G:\共用雲端硬碟\trash-sorter-data` 或 `G:\我的雲端硬碟\trash-sorter-data`），設成使用者環境變數，然後重開終端機：

   ```powershell
   setx TRASH_SORTER_DRIVE "G:\共用雲端硬碟\trash-sorter-data"
   ```

3. 確認：`Test-Path $env:TRASH_SORTER_DRIVE` 輸出 `True`。

git repo 本身固定放在 `%USERPROFILE%\Documents\repo\trash-sorter`，不在 Google Drive 或 OneDrive 的同步範圍內：同步 `.git` 會弄壞 repo。

## 資料夾結構

```
trash-sorter-data/
├── synthetic/<name>_<YYYY-MM-DD>.zip      合成資料集（YOLO-seg 格式，整包壓縮）
├── real/finetune/<YYYY-MM-DD>.zip         真實微調集
├── real_test/                             真實測試集：只分享給工作流 3 負責人與 PM
├── weights/<run>_<YYYY-MM-DD>/            best.pt + README.md
├── assets/raw/<類別>/                     3D 模型原始檔 + sizes.csv
├── videos/                                demo 影片
└── reports/                               報告、簡報素材
```

## 規則

- **只新增不覆寫**：新版本開新的檔名或資料夾（日期不同），舊的保留，實驗才能重現。
- **資料集整包壓成 zip** 再上傳：上千個小檔案同步很慢，也容易漏。
- **每次上傳附 README.md**（zip 則寫在同層的 `<name>.md`）：產生它的 commit（`git rev-parse --short HEAD`）、完整指令、關鍵指標（mAP50、成功率）。
- 上傳後在對應的 issue 或 PR 貼 Drive 連結。
- `real_test/` 在最終評估前不下載、不使用，見 [design.md](design.md#真實測試集由人完成程式提供工具)。

## 取用與上傳（agent 照這段做）

- 取用：從 `$TRASH_SORTER_DRIVE` 複製到 repo 的對應位置再解壓（`synthetic/*.zip` → `data/synthetic/`、`real/finetune/*.zip` → `data/real/finetune/`、`weights/<run>/best.pt` → `runs/segment/<run>/weights/`）。訓練與評估讀本機副本，不直接讀 Drive 路徑。
- 上傳：壓縮或複製到 `$TRASH_SORTER_DRIVE` 下對應的資料夾，用新名稱，連同 README 一起放，然後把路徑告訴使用者，請使用者在 issue 或 PR 貼連結。
- `TRASH_SORTER_DRIVE` 沒設定或路徑不存在時，停下來請使用者先完成上面的本機設定。
