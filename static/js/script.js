// Enhance upload page interactions (no inline JS to satisfy CSP)
document.addEventListener('DOMContentLoaded', () => {
	console.log('UI ready');
	const dz = document.getElementById('dropzone');
	const input = document.getElementById('resumes');
	const list = document.getElementById('file-list');
	const form = document.getElementById('upload-form');
	const bar = document.getElementById('progress-bar');

	if (!dz || !input) return; // Not on upload page

	const renderFiles = (files) => {
		if (!list) return;
		list.innerHTML = '';
		Array.from(files || []).forEach((f) => {
			const tag = document.createElement('span');
			tag.className = 'chip';
			tag.textContent = f.name;
			list.appendChild(tag);
		});
	};

	// Click anywhere in the dropzone to open native picker
	dz.addEventListener('click', () => input.click());

	// Drag & drop support
	dz.addEventListener('dragover', (e) => {
		e.preventDefault();
		dz.classList.add('over');
	});
	dz.addEventListener('dragleave', () => dz.classList.remove('over'));
	dz.addEventListener('drop', (e) => {
		e.preventDefault();
		dz.classList.remove('over');
		if (e.dataTransfer && e.dataTransfer.files) {
			// Assign dropped files to the hidden input so the form submits them
			input.files = e.dataTransfer.files;
			renderFiles(input.files);
		}
	});

	// When user browses and picks files
	input.addEventListener('change', () => renderFiles(input.files));

	// Lightweight progress hint during submit (purely visual)
	if (form && bar) {
		form.addEventListener('submit', () => {
			let i = 0;
			const t = setInterval(() => {
				i = Math.min(100, i + 4);
				bar.style.width = i + '%';
				if (i === 100) clearInterval(t);
			}, 60);
		});
	}
});

// Dashboard interactions: search/sort and selection banner (non-persistent)
document.addEventListener('DOMContentLoaded', () => {
	const search = document.getElementById('search');
	const sort = document.getElementById('sort');
	const container = document.getElementById('cards');
	const cards = container ? Array.from(container.querySelectorAll('.col')) : [];
	const banner = document.getElementById('selection-banner');
	const checkboxes = Array.from(document.querySelectorAll('.select-cb'));

	if (!container) return; // Not on dashboard

	const updateBanner = () => {
		const n = checkboxes.filter((cb) => cb.checked).length;
		if (banner) {
			if (n > 0) {
				banner.textContent = `${n} candidates selected.  Bulk actions`;
				banner.classList.remove('d-none');
			} else {
				banner.classList.add('d-none');
			}
		}
	};
	checkboxes.forEach((cb) => cb.addEventListener('change', updateBanner));

	const normalize = (t) => (t || '').toLowerCase();
	const applyFilters = () => {
		const q = normalize(search && search.value);
		cards.forEach((card) => {
			const text = normalize(card.textContent);
			card.style.display = !q || text.includes(q) ? '' : 'none';
		});
		const visible = cards.filter((c) => c.style.display !== 'none');
		if (sort && sort.value === 'name') {
			visible.sort((a, b) => {
				const an = a.querySelector('.card-title')?.textContent || '';
				const bn = b.querySelector('.card-title')?.textContent || '';
				return an.localeCompare(bn);
			});
		} else {
			visible.sort((a, b) => {
				const av = Number(a.querySelector('.ring')?.getAttribute('data-score') || 0);
				const bv = Number(b.querySelector('.ring')?.getAttribute('data-score') || 0);
				return bv - av;
			});
		}
		visible.forEach((el) => container.appendChild(el));
	};

	if (search) search.addEventListener('input', applyFilters);
	if (sort) sort.addEventListener('change', applyFilters);
	applyFilters();
});