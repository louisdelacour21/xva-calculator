// Pyodide in a worker: the computation (≈ 15 s at start, a few seconds per change) does not freeze the page.
importScripts('https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js');
const FILES = ['src/xva/__init__.py', 'src/xva/demo.py', 'src/xva/exposure.py', 'src/xva/hw1f.py', 'src/xva/market.py',
  'src/xva/products.py', 'src/xva/risk.py', 'src/xva/xva.py', 'market/spreads.yaml', 'market/swaption_vols.yaml',
  'portfolios/demo.yaml', 'web/xva_web.py'];
const say = (type, data) => postMessage({ type, data });

const ready = (async () => {
  say('step', 'Loading Python (Pyodide)…');
  const py = await loadPyodide();
  say('step', 'Loading numpy, scipy, pandas…');
  await py.loadPackage(['numpy', 'scipy', 'pandas', 'pyyaml']);
  for (const f of FILES) {
    const r = await fetch('../' + f);
    if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
    py.FS.mkdirTree('/home/pyodide/XVA/' + f.split('/').slice(0, -1).join('/'));
    py.FS.writeFile('/home/pyodide/XVA/' + f, await r.text());
  }
  say('step', 'Calibration, simulation of 10,000 paths and MtM cube…');
  py.runPython("import sys, json; sys.path[:0] = ['/home/pyodide/XVA/src', '/home/pyodide/XVA/web']; import xva_web");
  say('ready', JSON.parse(py.runPython('json.dumps(xva_web.start())')));
  return py;
})().catch((e) => { say('error', String(e)); throw e; });

onmessage = async ({ data }) => {
  const py = await ready;
  py.globals.set('_p', JSON.stringify(data));
  try { say('result', JSON.parse(py.runPython('json.dumps(xva_web.compute(json.loads(_p)))'))); }
  catch (e) { say('error', String(e)); }
};
