/* main.js – SpamShield AI client-side logic */

/* ─── Tab Switching ─────────────────────────────────── */
function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
    btn.setAttribute('aria-selected', 'false');
  });
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

  const btn   = document.getElementById(`tab-${tabName}`);
  const panel = document.getElementById(`panel-${tabName}`);
  if (btn)   { btn.classList.add('active'); btn.setAttribute('aria-selected', 'true'); }
  if (panel) { panel.classList.add('active'); }
}

/* ─── Character Counter ──────────────────────────────── */
const emailTextarea = document.getElementById('email_text');
const charCounter   = document.getElementById('char-counter');

if (emailTextarea && charCounter) {
  const maxLen = parseInt(emailTextarea.getAttribute('maxlength')) || 5000;
  emailTextarea.addEventListener('input', () => {
    const len = emailTextarea.value.length;
    charCounter.textContent = `${len} / ${maxLen} characters`;
    charCounter.style.color = len > maxLen * 0.9 ? '#f97316' : '';
  });
}

/* ─── Example Filler ─────────────────────────────────── */
const examples = {
  spam: `Congratulations! You've been selected as a lucky winner of our exclusive $10,000 cash prize! Click the link below to CLAIM NOW before it expires. Limited time offer – act fast! Free iPhone 16 Pro Max waiting for you. Call 1-800-WIN-NOW. This is NOT a scam. 100% guaranteed.`,
  ham: `Hi team, just a reminder that our project sync meeting is scheduled for tomorrow at 10 AM in Conference Room B. Please review the attached agenda and come prepared with updates on your current tasks. Let me know if you have any conflicts. Thanks, Ankit`
};

function fillExample(type) {
  const ta = document.getElementById('email_text');
  if (!ta) return;
  ta.value = examples[type] || '';
  ta.dispatchEvent(new Event('input'));
  ta.focus();
  ta.scrollTop = 0;
}

/* ─── Clear Form ─────────────────────────────────────── */
function clearForm(fieldId) {
  const el = document.getElementById(fieldId);
  if (el) { el.value = ''; el.dispatchEvent(new Event('input')); el.focus(); }
}

/* ─── Form Submit Loader ─────────────────────────────── */
const singleForm = document.getElementById('singleForm');
if (singleForm) {
  singleForm.addEventListener('submit', function() {
    const btn    = document.getElementById('submitBtn');
    const text   = btn && btn.querySelector('.btn-text');
    const loader = btn && btn.querySelector('.btn-loader');
    if (text)   text.hidden   = true;
    if (loader) loader.hidden = false;
    if (btn)    btn.disabled  = true;
  });
}

/* ─── Copy Code Block ────────────────────────────────── */
function copyCode(btn) {
  const code = btn.getAttribute('data-code');
  if (code) {
    navigator.clipboard.writeText(code).then(() => {
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
    });
  }
}

/* ─── Copy Analyzed Text ─────────────────────────────── */
function copyText(elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;
  navigator.clipboard.writeText(el.innerText).then(() => {
    const btn = el.parentElement.querySelector('.copy-btn');
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = '✅ Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 2000);
    }
  });
}

/* ─── Mobile Nav Toggle ──────────────────────────────── */
const navToggle = document.getElementById('navToggle');
const navLinks  = document.querySelector('.nav-links');

if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => {
    const isOpen = navLinks.classList.toggle('open');
    navToggle.setAttribute('aria-expanded', isOpen.toString());
  });
  // Close nav when link clicked
  navLinks.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => { navLinks.classList.remove('open'); navToggle.setAttribute('aria-expanded', 'false'); });
  });
}

/* ─── Animate Gauge on Page Load ────────────────────── */
window.addEventListener('load', () => {
  document.querySelectorAll('.gauge-fill, .prob-bar, .conf-bar, .mini-bar').forEach(el => {
    const target = el.style.width;
    el.style.width = '0%';
    requestAnimationFrame(() => requestAnimationFrame(() => { el.style.width = target; }));
  });
});

/* ─── Auto-dismiss alerts ────────────────────────────── */
document.querySelectorAll('.alert').forEach(alert => {
  setTimeout(() => {
    alert.style.transition = 'opacity 0.5s ease';
    alert.style.opacity = '0';
    setTimeout(() => alert.remove(), 500);
  }, 4000);
});
