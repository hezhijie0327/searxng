# HANDOVER — zjsearch 开发现状与待办

> 更新：2026-09-14。供后续开发会话（AI 或人类）快速接手。
> 架构与约定见 `AGENTS.md` 与 `client/zjsearch/README.rst`。

## 当前状态

- 分支 `skin`：已 rebase 到 `origin/master`，本地领先 1 个提交（尚未推送）。
- 主题（`client/zjsearch`）重构为四层结构：`lib → components → features → pages`，
  依赖只向下；导入统一走 `@/` 别名；biome + tsc + vite build 全绿。
- 未做：vitest 测试设施（`layout.ts`、`searchParams.ts`、`usePreferencesForm`
  是最值得先测的三个纯模块）；`components/Dropdown` 与 preferences 里的
  `Select` 两套下拉实现尚未统一（已知遗留，不阻塞）。

## 最近新增（均已浏览器实测：桌面 1440 + 移动 390）

1. **结果页 meta 行**（`features/results/DebugPanels.tsx`，唯一的行拥有者）：
   - 「找到 N 条相关结果 ▾」计数本身是开关，展开操作条：
     `复制链接`（复制可分享 URL，点击变「已复制」）+ 格式 chips（带图标，
     清单来自 `globals.search_formats`，服务端已排除 html）。
   - 「耗时/引擎消息 ▾」开关不变，与结果条互斥展开。
   - 分享 URL 统一来自 `lib/searchParams.ts` 的 `shareableSearchUrl(data)`。
2. **偏好设置第七个 tab「信息」**（`pages/preferences/tabs/AboutTab.tsx`）：
   挂载时抓取 `globals.about_url` 的 page-data，复用 embedded `InfoPage`。
3. **About 页页脚整合**（`pages/InfoPage.tsx` 的 `AboutFooter`，仅 about 页）：
   实例卡（`/config`：版本 + 引擎/分类/插件计数）+ 「功能来自 SearXNG」署名 +
   许可证链接（openDocument 打开 LICENSE.txt）+ 合规链接（隐私政策/联系/公共实例，
   按配置条件渲染）。偏好设置自己的旧页脚已删除。
4. **首页 `custom_links` chips**（`IndexPage.tsx`）：settings.yml 配置的
   自定义链接渲染为搜索框下方 chips，未配置时零渲染。
5. **i18n**：`lib/i18n/` 每语言一文件；`en.ts` 的 `StringKey` 类型化所有
   key；本轮新增 `copy_search_url / copy_link / privacypolicy / contact /
   public_instances / about_instance / categories / plugins`。

## 开放问题（Windows 开发环境，非实现 bug）

**症状**：Windows 本地实例（granian + WhiteNoise，端口 8888）上 45 个
`/static/chunk/*.min.js` 中恰有 2 个返回 404（HTML 兜底页）：
`BpgPcn_G.min.js`（偏好设置 chunk）与 `by2rA14n.min.js`。其余 43 个 200。
磁盘上文件齐全且内容为最新构建（os.link 硬链接）。**后果**：偏好设置抽屉
当前在这台机器上可能无法打开（chunk 导入失败）。

**已排除**：文件不存在；junction 方案（WhiteNoise 的目录遍历会跳过 junction，
已改回逐文件硬链接）；浏览器缓存（cache-busting 后复现）。

**主要怀疑**：AGENTS.md 已记载的孤儿 worker 机制——granian worker 经
SO_REUSEADDR 共享绑定 :8888，旧 worker 的 WhiteNoise 索引里没有这两个
新 chunk 名，按连接分配到旧 worker 时即 404。我在 17:44 的"杀净 5 个
监听者再单启"实验后仍然复现，说明可能还有未被 netstat 第一行枚举出的
进程（或 WhiteNoise 索引时机问题）。

**后续处理建议**：
1. 彻底按进程名清理（不只按端口）：
   `taskkill //IM granian.exe //F` + `taskkill //IM python.exe //F`，
   确认 `netstat -ano | grep :8888` 无监听后单启一个实例，重测 45 个 URL。
2. 若仍复现：在 `searx/webapp.py` 的 WhiteNoise 初始化处临时打印索引内容，
   确认是索引缺失还是路由回落。
3. Linux 部署不受影响（无硬链接/junction 变通，路径映射正常）。

## Windows 资产服务现状（每次构建后必做）

`zjsearch.min.js/css` 与整个 `chunk/` 目录都靠硬链接进 `searx/static/`。
构建后执行（Git Bash，仓库根目录）：

```sh
cd searx/static
rm -f zjsearch.min.js zjsearch.min.css
cmd //c "mklink /H zjsearch.min.js themes\\zjsearch\\zjsearch.min.js"
cmd //c "mklink /H zjsearch.min.css themes\\zjsearch\\zjsearch.min.css"
python - <<'EOF'
import os, shutil
src, dst = os.path.join("themes","zjsearch","chunk"), "chunk"
shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
for f in os.listdir(src): os.link(os.path.join(src,f), os.path.join(dst,f))
EOF
```

注意：junction（`mklink /J`）不可用——WhiteNoise 的目录遍历会跳过它；
`mklink /H` 的 Git Bash 反斜杠转义不可靠——用上面的 Python `os.link`。
症状识别：`curl /static/chunk/<名>` 返回 HTML 即为失效（正常应返回 JS）。
