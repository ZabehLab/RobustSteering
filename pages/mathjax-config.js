// Pinned local MathJax bundle: equations render without a runtime CDN dependency.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    packages: ["base", "ams", "newcommand", "configmacros", "color"],
    macros: {
      statevar: ["{\\color[RGB]{0,124,124}#1}", 1],
      steervar: ["{\\color[RGB]{139,30,30}#1}", 1],
      xk: "\\statevar{x_k}",
      xkp: "\\statevar{x_{k+1}}",
      Ak: "\\statevar{A_k}",
      Dk: "\\statevar{D_k}",
      wk: "\\statevar{w_k}",
      uk: "\\steervar{u_k}",
      Bk: "\\steervar{B_k}",
      Kk: "\\steervar{K_k}",
      zk: "\\steervar{z_k}",
      gammastar: "\\steervar{\\gamma^\\star}",
      Srob: "\\steervar{S_{\\mathrm{rob}}}"
    }
  },
  loader: {
    load: ["[tex]/color"],
    source: {"[tex]/color": "color.js"}
  },
  svg: {fontCache: "local"},
  options: {
    enableMenu: false,
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};
