# Paper project page

Custom, buildless HTML/CSS/JavaScript. Local development only; no publishing workflow has been added.

## Preview

From the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory pages
```

Open `http://127.0.0.1:8765/`. No npm installation or build step is required.

## Structure

- `index.html`: paper metadata, section order, abstract, overall results, and citation.
- `styles.css`: responsive layout and a single accent token, `--teal: #007C7C`.
- `site.js`: figure enlargement and citation copying. A single non-sticky contents row sits below the abstract, without an Animation link. The untitled interactive-example area remains reserved for a later iteration.
- `mathjax-config.js` and `assets/vendor/mathjax/`: locally served MathJax 3.2.2 with its color extension and Apache-2.0 license. Semantic macros match the manuscript's teal state/model terms (`#007C7C`) and red steering/control terms (`#8B1E1E`). No runtime CDN or font download is required for this page's equations.
- `assets/`: the supplied PDF, unmodified final figures copied from the Overleaf bundle, table source, and the custom favicon.
- `content-source.json`: manuscript and asset provenance.

## Content source

The authoritative content source is `paper/ICRL2027/arXiv_AppleStyle.tex` from the user-supplied October 10 Overleaf bundle. The abstract, introduction, method, and results follow that file; alternate drafts and the personal `rudra.tex` notebook are not website sources. The abstract is retained verbatim from the initial webpage. Other copy selects or condenses the paper's text, adapting manuscript cross-references and TeX notation for the webpage. Numeric summaries retain the means in its main overview table, labeled `tab:main-results-summary`, with grouped condition headers and existing reported uncertainties. TruthfulQA and HarmBench show Baseline and H-infinity only. Original values were checked against the final paper tables and then removed from those two website tables at the user's request; their audit record remains in `content-source.json`. “Baseline” retains the overview table's strongest non-H-infinity result, including Original. Per-cell values and uncertainty sources are recorded in `assets/table-values.json`; these use the full paper tables and, for aggregate HarmBench/MGSM uncertainties omitted from the compact overview, existing canonical generated summary tables. No new uncertainty calculation is performed. The overview table's original TeX is retained in `assets/overview-table.tex`. The current PDF remains in the assets for reference. At the user's request, the highlighted arXiv button temporarily links to GitHub for visual review; its tooltip and source comment identify this placeholder. Replace its destination with the real arXiv URL before publication. No arXiv identifier or release claim has been invented.

Figure files are byte-identical copies of the `Final_Figures/` assets referenced by `arXiv_AppleStyle.tex`, not PDF extractions, replots, or modified benchmark images. Existing figure colors are preserved. Site typography is sans-serif; the interface uses black, white, and the exact paper teal, `StateTeal` (`#007C7C`). The interactive-example area is visibly marked as a layout placeholder.

The GitHub resource button links to the repository origin, `https://github.com/rgs2151/controller`. Its icon is Primer's `mark-github-16` SVG, distributed with the accompanying MIT license in `assets/github-LICENSE`. The unmodified white arXiv wordmark comes from [arXiv's official brand site](https://info.arxiv.org/brand/logos.html), specifically `https://info.arxiv.org/assets/arxiv-logo-one-color-white.svg`. MathJax's combined TeX/SVG component is documented in [MathJax's official guide](https://docs.mathjax.org/en/v3.2/web/components/combined.html).

Typography uses Arial first, with Helvetica/sans-serif fallbacks. The landing and abstract are retained. Other sections use a shared 16px body/caption/table size, 14px supporting labels, 26px benchmark headings, and 28px section headings, with responsive heading sizes on smaller screens. Motivation restores the exact earlier three-paragraph copy with large teal numbers beside each statement, without bold lead-ins. Section spacing and figure/table gaps follow a shared scale. Tables use their natural content width rather than stretching to the figure width; two-level headers separate conditions or metrics from the methods being compared. Math uses MathJax's TeX glyphs; citation code uses a monospace face.

## Eventual GitHub Pages hosting

The existing repository documentation remains untouched in `docs/`. This site is isolated in lowercase `pages/`. GitHub's branch-based publishing supports the repository root or `docs/`; publishing only `pages/` can instead use an Actions workflow that uploads this folder. That workflow and any deployment configuration will be added only when publication is approved.

Official reference: [Configuring a publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
