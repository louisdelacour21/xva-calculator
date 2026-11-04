// Pyodide dans un worker : le calcul (≈ 15 s au démarrage, quelques secondes par réglage) ne fige pas la page.
importScripts('https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js');
const FILES = ['src/xva/__init__.py', 'src/xva/demo.py', 'src/xva/exposure.py', 'src/xva/hw1f.py', 'src/xva/market.py',
  'src/xva/products.py', 'src/xva/risk.py', 'src/xva/xva.py', 'market/spreads.yaml', 'market/swaption_vols.yaml',
  'portfolios/demo.yaml', 'web/xva_web.py'];
const say = (type, data) => postMessage({ type, data });

const ready = (async () => {
  say('etape', 'Chargement de Python (Pyodide)…');
  const py = await loadPyodide();
  say('etape', 'Chargement de numpy, scipy, pandas…');
  await py.loadPackage(['numpy', 'scipy', 'pandas', 'pyyaml']);
  for (const f of FILES) {
    const r = await fetch('../' + f);
    if (!r.ok) throw new Error(`${f} : HTTP ${r.status}`);
    py.FS.mkdirTree('/home/pyodide/XVA/' + f.split('/').slice(0, -1).join('/'));
    py.FS.writeFile('/home/pyodide/XVA/' + f, await r.text());
  }
  say('etape', 'Calibration, simulation de 10 000 chemins et cube MtM…');
  py.runPython("import sys, json; sys.path[:0] = ['/home/pyodide/XVA/src', '/home/pyodide/XVA/web']; import xva_web");
  say('pret', JSON.parse(py.runPython('json.dumps(xva_web.demarrer())')));
  return py;
})().catch((e) => { say('erreur', String(e)); throw e; });

onmessage = async ({ data }) => {
  const py = await ready;
  py.globals.set('_p', JSON.stringify(data));
  try { say('resultat', JSON.parse(py.runPython('json.dumps(xva_web.calculer(json.loads(_p)))'))); }
  catch (e) { say('erreur', String(e)); }
};
