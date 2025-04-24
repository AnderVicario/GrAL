// === Hizkuntza menuaren bistaratzea ===
const langToggle = document.getElementById('langToggle');
const langDropdown = document.getElementById('langDropdown');

langToggle.addEventListener('click', () => {
  langDropdown.classList.toggle('hidden');
});

document.addEventListener('click', (e) => {
  if (!langDropdown.contains(e.target) && !langToggle.contains(e.target)) {
    langDropdown.classList.add('hidden');
  }
});

// === Modu aurreratuaren aktibazioa ===
let isAdvancedMode = false;
const advancedToggle = document.getElementById('advancedToggle');
const advancedIcon   = document.getElementById('advancedIcon');

advancedToggle.addEventListener('click', () => {
  isAdvancedMode = !isAdvancedMode;
  advancedIcon.classList.toggle('text-[#607AFB]', isAdvancedMode);
  advancedIcon.classList.toggle('text-[#D5D6DD]', !isAdvancedMode);
});

// === Inprimakia bidali eta karga‐ikonoa erakutsi ===
const form          = document.querySelector('form');
const submitBtn     = document.getElementById('submitButton');
const submitText    = document.getElementById('submitText');
const loadingSpinner = document.getElementById('loadingSpinner');

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const input = form.querySelector('input[name="user_input"]').value.trim();
  if (!input) return;

  // biraka‐ikonoa erakutsi
  submitText.classList.add('hidden');
  loadingSpinner.classList.remove('hidden');
  submitBtn.disabled = true;
  submitBtn.classList.add('opacity-75');

  const data = new FormData(form);
  data.append('advanced_mode', isAdvancedMode);

  fetch(form.action, {
    method: form.method,
    body: data
  })
  .then(res => {
    if (res.ok) {
      form.reset();
      window.location.reload();
    }
  })
  .catch(err => {
    console.error(err);
    submitText.classList.remove('hidden');
    loadingSpinner.classList.add('hidden');
    submitBtn.disabled = false;
    submitBtn.classList.remove('opacity-75');
  });
});

// === Fitxategien aurreikuspena ===
function handleFileSelect(e) {
  const files = e.target.files;
  const fileListDiv = document.getElementById('fileList');

  if (files.length) {
    let html = '';
    for (const f of files) {
      html += `
        <div class="flex items-center gap-2 mb-1">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" …>…</svg>
          <span class="truncate">${f.name}</span>
        </div>
      `;
    }
    fileListDiv.innerHTML = html;
    fileListDiv.classList.remove('hidden');
  } else {
    fileListDiv.classList.add('hidden');
  }
}

document.getElementById('fileInput').addEventListener('change', handleFileSelect);

// === Karuselaren logika ===
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.carousel-prev, .carousel-next');
  if (!btn) return;

  const container = btn.closest('.carousel-container');
  const items = Array.from(container.querySelectorAll('.carousel-item'));
  const current = items.findIndex(i => i.classList.contains('active'));
  const dir = btn.classList.contains('carousel-prev') ? -1 : +1;
  const next = Math.min(Math.max(current + dir, 0), items.length - 1);

  if (next !== current) {
    items[current].classList.replace('active', 'hidden');
    items[next].classList.replace('hidden', 'active');

    // botoien egoera eguneratu
    container.querySelector('.carousel-prev')
      .classList.toggle('!invisible', next === 0);
    container.querySelector('.carousel-next')
      .classList.toggle('!invisible', next === items.length - 1);
  }
});

function initCarousels() {
  document.querySelectorAll('.carousel-container').forEach(c => {
    const items = c.querySelectorAll('.carousel-item');
    items.forEach((it, i) => {
      it.classList.toggle('active', i === 0);
      it.classList.toggle('hidden',  i !== 0);
    });
    c.querySelector('.carousel-prev')?.classList.add('!invisible');
    c.querySelector('.carousel-next')
     .classList.toggle('!invisible', items.length <= 1);
  });
}

// orria kargatzean eta txata eguneratzean exekutatu
['DOMContentLoaded', 'chatUpdate'].forEach(evt =>
  document.addEventListener(evt, initCarousels)
);
