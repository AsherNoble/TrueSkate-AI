// Browser control is scoped to the operator-requested isolated rig Chrome.
const {chromium} = require('playwright-core');
const fs = require('node:fs');
(async () => {
  let action = JSON.parse(process.argv[2] || '{"kind":"state"}');
  const browser = await chromium.connectOverCDP('http://127.0.0.1:19222');
  const context = browser.contexts()[0];
  const pages = context.pages();
  if (action.kind === 'state' && !pages.length) {
    console.log(JSON.stringify({pages:[],contexts:browser.contexts().length}));
    process.exit(0);
  }
  const page = action.kind === 'new' ? await context.newPage() : action.target_prefix ? pages.find(p=>p.url().startsWith(action.target_prefix)) : action.target_url ? pages.find(p=>p.url()===action.target_url || p.url().split("#")[0]===action.target_url) : pages[action.page || 0];
  if (!page) throw new Error('observed target page is no longer open');
  const steps = action.steps || [action];
  for (const step of steps) {
  action = step;
  if (action.kind === 'new') await page.goto(action.url,{waitUntil:'domcontentloaded'});
  if (action.kind === 'click') { let loc=page.getByRole(action.role || 'button', {name:action.name,exact:action.exact !== false}); if(action.nth!==undefined)loc=loc.nth(action.nth); await loc.click(); }
  if (action.kind === 'dblclick') await page.getByText(action.name,{exact:true}).first().dblclick();
  if (action.kind === 'fill') await page.getByRole(action.role || 'textbox', {name:action.name,exact:true}).fill(action.value);
  // Fills a secret from a local file so its value never appears in arguments or output.
  if (action.kind === 'fill_file') await page.locator(action.selector).fill(fs.readFileSync(action.file, 'utf8').trim());
  if (action.kind === 'text') await page.getByText(action.name,{exact:true}).click();
  if (action.kind === 'goto') await page.goto(action.url, {waitUntil:'domcontentloaded'});
  if (action.kind === 'screenshot') await page.screenshot({path:action.path,fullPage:false});
  if (action.kind === 'key') await page.keyboard.press(action.key);
  if (action.kind === 'frames') console.log(JSON.stringify(await Promise.all(page.frames().map(async f=>({url:f.url(),text:await f.locator('body').innerText({timeout:1000}).catch(()=>null)})))));
  if (action.kind === 'dom') console.log(JSON.stringify(await page.locator(action.selector).evaluateAll(es=>es.map(e=>({tag:e.tagName,type:e.getAttribute('type'),role:e.getAttribute('role'),name:e.getAttribute('aria-label'),html:e.outerHTML.slice(0,1500)})))));
  if (action.kind === 'files') {
    const cdp = await context.newCDPSession(page);
    const {root} = await cdp.send('DOM.getDocument');
    const {nodeId} = await cdp.send('DOM.querySelector', {nodeId:root.nodeId,selector:action.selector});
    if (!nodeId) throw new Error('observed upload input is no longer present');
    await cdp.send('DOM.setFileInputFiles',{nodeId,files:action.files});
  }
  }
  console.log(JSON.stringify({pages:await Promise.all(context.pages().map(async p=>({url:p.url().startsWith("https://accounts.google.com/")?p.url().split("?")[0]:p.url(),title:await p.title()}))),
    activeUrl:page.url().startsWith("https://accounts.google.com/")?page.url().split("?")[0]:page.url(), text:(await page.locator('body').innerText({timeout:10000})).split('\n').filter(l=>!/^['\s]*[A-Za-z0-9+/=]{80,}/.test(l)).map(l=>l.length>1400?l.slice(0,300)+' ... [long code line] ... '+l.slice(-100):l).join('\n'),
    controls:await page.locator('button,input,a,select,[role="button"],[role="textbox"],[role="menuitem"],[role="dialog"],[role="tab"],[contenteditable="true"]').evaluateAll(els=>els.filter(e=>e.getClientRects().length).map(e=>({tag:e.tagName,role:e.getAttribute('role'),name:e.getAttribute('aria-label') || e.innerText || e.getAttribute('placeholder'),type:e.getAttribute('type'),href:e.getAttribute('href')})).filter(e=>e.name!=='More actions').slice(0,90))}));
  process.exit(0); // Drop client transport without closing the rig browser.
})().catch(e=>{console.error(e.message);process.exit(1)});
