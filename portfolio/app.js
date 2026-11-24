const $ = id => document.getElementById(id);
const names = {
  late_fusion: 'Late fusion',
  image: 'Image-only baseline',
  concat_drop: 'Matched concatenation',
  gated: 'Gated fusion',
  text: 'Text-only baseline',
};
const score = value => value.toFixed(4);
const percent = value => (value * 100).toFixed(1) + '%';
const format = new Intl.NumberFormat();
let sample = false;
let previewUrl;
let version = 0;

function node(tag, value, name) {
  const element = document.createElement(tag);
  element.textContent = value;
  if (name) element.className = name;
  return element;
}
function showError(message) {
  $('status').textContent = message;
  $('status').classList.add('error');
}
function resetResult() {
  version += 1;
  $('result').hidden = true;
  $('status').textContent = '';
  $('status').classList.remove('error');
}
function showImage(url) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = url.startsWith('blob:') ? url : undefined;
  $('preview').src = url;
  $('preview').hidden = false;
}
function compare(report) {
  const condition = $('condition').value;
  $('scores').replaceChildren(...Object.entries(names).map(([key, name]) => {
    const rows = report.rows.filter(row => row.model === key && row.condition === condition);
    const value = rows.reduce((sum, row) => sum + row.f1, 0) / rows.length;
    const bar = document.createElement('progress');
    bar.max = 1;
    bar.value = value;
    bar.setAttribute('aria-label', name + ' macro-F1');
    const line = node('div', '', 'bar');
    line.append(node('span', name), bar, node('strong', score(value)));
    return line;
  }));
}
function setUp(report) {
  $('mode').textContent = report.live ? 'Verified local checkpoints' : 'Recorded evaluation';
  $('test-size').textContent = format.format(report.test_size) + ' test rows';
  $('label-audit').textContent = format.format(report.label_disagreements) + ' mirror-label disagreements were excluded from target selection.';
  const test = report.test;
  $('group-test').textContent = 'Prespecified clean ensemble comparison: gated − matched concatenation ' +
    score(test.difference_macro_f1) + ' macro-F1; 95% interval [' + test.bootstrap_ci95.map(score).join(', ') + '], p = ' +
    score(test.permutation_p_two_sided) + '. The interval includes zero.';
  $('condition').addEventListener('change', () => compare(report));
  compare(report);
  if (!report.live) return;
  $('offline').hidden = true;
  $('form').hidden = false;
  if (typeof report.example_caption === 'string') {
    $('sample').hidden = false;
    $('sample').addEventListener('click', () => {
      resetResult();
      sample = true;
      $('image').value = '';
      $('caption').value = report.example_caption;
      showImage('/example.jpg');
    });
  }
}
function imageBytes(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = () => reject(new Error('Unable to read the image.'));
    reader.readAsDataURL(file);
  });
}
$('image').addEventListener('change', () => {
  resetResult();
  sample = false;
  const file = $('image').files[0];
  $('preview').hidden = true;
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) {
    showError('Choose an image up to 8 MiB.');
    $('image').value = '';
    return;
  }
  showImage(URL.createObjectURL(file));
});
$('caption').addEventListener('input', resetResult);
$('form').addEventListener('submit', async event => {
  event.preventDefault();
  resetResult();
  const current = version;
  const button = event.submitter;
  button.disabled = true;
  $('status').textContent = 'Running local models…';
  try {
    const body = {text: $('caption').value};
    if (sample) body.sample = true;
    else {
      const file = $('image').files[0];
      if (!file) throw new Error('Choose an image or load the local example.');
      body.image = await imageBytes(file);
    }
    const response = await fetch('/api/predict', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body),
    });
    const value = await response.json();
    if (current !== version) return;
    if (!response.ok) throw new Error(value.error);
    const cards = [
      ['Gated fusion', value.probability_informative],
      ['Late fusion', value.comparison.late_fusion],
      ['Image only', value.comparison.image_only],
    ].map(([name, probability]) => {
      const card = node('div', '', 'metric');
      card.append(node('small', name), node('strong', percent(probability)),
        node('small', probability > .5 ? 'Informative' : 'Not informative'));
      return card;
    });
    $('text-state').textContent = value.text_status === 'used' ? 'Caption features used. Scores indicate image informativeness.' :
      value.text_status === 'unrecognized' ? 'No vocabulary matches. Text was masked and late fusion used the image-only score.' :
      'Caption omitted. Late fusion uses the image-only score.';
    $('predictions').replaceChildren(...cards);
    $('result').hidden = false;
    $('status').textContent = 'Comparison ready.';
  } catch (error) {
    if (current === version) showError(error.message);
  } finally {
    button.disabled = false;
  }
});
fetch('/api/study').then(response => {
  if (!response.ok) throw new Error('Unable to load evaluation.');
  return response.json();
}).then(setUp).catch(error => {
  $('failure').textContent = error.message;
  $('failure').hidden = false;
});
