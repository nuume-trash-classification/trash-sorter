# 協作規則

4 位組員 + 學長，指導老師不在 GitHub 上。這頁說明工作怎麼分、怎麼排進度、程式碼怎麼進 `main`。

## 角色

| 角色 | 負責 | GitHub 權限 |
|---|---|---|
| PM（專案管理） | 維護看板、主持例會、寫給老師的進度報告 | org Owner |
| 工作流負責人（每人一個） | 追自己工作流的 issue 有沒有在動；不代表只能做自己的工作流 | team `members`（Write） |
| 學長（kevin00156） | 技術顧問、review PR | org Owner |
| 指導老師 | 看進度報告，做需要老師拍板的決定 | 不在 GitHub 上 |

新組員：org Owner 邀請進 `nuume-trash-classification` 後，把對方加進 team `members` 才能推分支、開 PR。

### 給老師的進度報告

老師不看 GitHub，由 PM 每次迭代結束（或每月）整理一份報告寄給老師：
Roadmap 截圖、這期完成的 milestone / issue、指標變化（取物成功率、mAP50），以及需要老師決定的事
（看板上標 `teacher` label 的 issue）。老師的回覆由 PM 記回對應 issue 或會議紀錄。

四個工作流的內容見 [README「四個工作流」](README.md#四個工作流每人一個)。
issue 用 label 標工作流：`area:assets`、`area:training`、`area:real-data`、`area:eval-demo`、`area:infra`。

## 進度管理

- **所有工作都開 issue**，寫清楚「完成條件」。沒有 issue 的工作，別人看不到。
- **看板**（GitHub Projects）：`Backlog → Todo → In Progress → In Review → Done`。
  開始做就把卡片拉到 In Progress，並把自己設成 Assignee。
- **Milestone** = 專題的時程節點。每個 issue 掛到一個 milestone。
- **兩週一次迭代**：開始時從 Backlog 挑這兩週要做的；結束時 demo + 回顧（約 30 分鐘）。
- **每週例會 15 分鐘**：每人講「做完什麼、下週做什麼、卡在哪」。紀錄放 [docs/meetings/](docs/meetings/)。
- 重要技術決定寫成 ADR，放 [docs/adr/](docs/adr/)。

## 程式碼進 `main` 的流程

`main` 有保護：**不能直接 push，一律走 Pull Request（PR）**，CI 要過、至少 1 人 review。

第一次設定（只做一次）：

```bat
git clone https://github.com/nuume-trash-classification/trash-sorter.git
cd trash-sorter
uv sync
```

每做一件事：

```bat
:: 1. 從最新的 main 開分支，名稱帶 issue 編號
git switch main
git pull
git switch -c 12-convert-glass-assets

:: 2. 改程式，提交前檢查
uv run ruff check --fix .
uv run ruff format .
uv run pytest

:: 3. 提交並推上去
git add <改到的檔案>
git commit -m "Convert glass bottle assets to USD"
git push -u origin 12-convert-glass-assets

:: 4. 開 PR（描述裡寫 Closes #12，合併時 issue 會自動關閉）
gh pr create --fill
```

PR 被要求修改時：在同一個分支繼續 commit、push，PR 會自動更新。
合併用 **Squash and merge**，合併後可以刪掉分支。

### Review 看什麼

- 有沒有符合 issue 的完成條件。
- 有沒有違反「特權資訊規則」：`--perception yolo` 的流程不能讀模擬器裡物體的位置或類別
  （見 [docs/design.md](docs/design.md#特權資訊規則重要)）。
- 模擬 / 訓練的改動要附數字（取物成功率、mAP50），不要只寫「看起來比較好」。

## 不進 git 的東西

資料集、模型權重、渲染影片、轉出來的 USD 資產都不進 git（見 `.gitignore`）。
這些放團隊共用雲端硬碟，PR 或 issue 裡附連結即可。API key、`.env` 絕對不要 commit。

## 真實測試集隔離

`data/real/test`（約 300 張）**最後評估前誰都不碰**：不拿來訓練、不拿來調參數、不先偷看分數。
由工作流 3 負責人保管，最終評估只跑一次（見工作流 4）。
