import { useState } from 'react';

const metrics = [
  { key: 'calories', label: 'Calories', unit: 'kcal' },
  { key: 'protein_g', label: 'Protein', unit: 'g' },
  { key: 'carbs_g', label: 'Carbs', unit: 'g' },
  { key: 'fat_g', label: 'Fat', unit: 'g' },
  { key: 'fiber_g', label: 'Fiber', unit: 'g' }
];

const sourceLabels = {
  local_yolo: 'Local YOLO',
  local_yolo_primary: 'Indian YOLO',
  local_yolo_secondary: 'General YOLO'
};

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (selectedFile) => {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith('image/')) {
      setError('Please upload a valid image file.');
      return;
    }

    setFile(selectedFile);
    setError('');
    setAnalysis(null);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!file) {
      setError('Please select an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        if (data && data.meal_name && data.foods) {
          setAnalysis(data);
          return;
        }
        throw new Error(data.error || 'Something went wrong.');
      }

      setAnalysis(data);
    } catch (err) {
      setError(err.message || 'Unable to analyze the image.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(34,197,94,0.22),_transparent_45%),linear-gradient(135deg,_#020617,_#111827)] text-slate-100">
      <div className="mx-auto flex max-w-7xl flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8">
        <header className="rounded-3xl border border-emerald-500/20 bg-slate-900/70 p-8 shadow-2xl shadow-emerald-950/30 backdrop-blur">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <p className="mb-3 inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-sm font-medium text-emerald-300">
                <span className="text-lg">🥗</span>
                AI-powered food nutrition insights
              </p>
              <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
                Analyze meals in seconds with vision AI.
              </h1>
              <p className="mt-4 text-lg text-slate-300">
                Upload a meal photo, and the app will estimate portions and summarize the nutrition profile.
              </p>
            </div>
            <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 px-5 py-4 text-sm text-emerald-200">
              <p className="font-semibold">No API • No login • Local YOLO</p>
            </div>
          </div>
        </header>

        <div className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr]">
          <section className="rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-2xl shadow-slate-950/40 backdrop-blur">
            <form onSubmit={handleSubmit} className="space-y-6">
              <label
                htmlFor="image-upload"
                className="flex cursor-pointer flex-col items-center justify-center rounded-3xl border-2 border-dashed border-emerald-400/50 bg-slate-800/60 px-6 py-16 text-center transition hover:border-emerald-300 hover:bg-emerald-500/10"
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  handleFileChange(event.dataTransfer.files[0]);
                }}
              >
                <div className="mb-4 rounded-full bg-emerald-500/10 p-4 text-4xl">📸</div>
                <p className="text-xl font-semibold text-slate-100">Drop your food image here</p>
                <p className="mt-2 text-sm text-slate-400">PNG, JPG, and WEBP are supported</p>
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(event) => handleFileChange(event.target.files?.[0])}
                />
              </label>

              {previewUrl && (
                <div className="overflow-hidden rounded-3xl border border-slate-800 bg-slate-950/70">
                  <img src={previewUrl} alt="Uploaded meal preview" className="h-72 w-full object-cover" />
                </div>
              )}

              <button
                type="submit"
                className="flex w-full items-center justify-center gap-2 rounded-2xl bg-emerald-500 px-5 py-3 font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={loading || !file}
              >
                {loading ? 'Analyzing your meal...' : 'Analyze Meal'}
              </button>

              {error && <p className="rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">{error}</p>}
            </form>
          </section>

          <section className="space-y-6">
            {loading && (
              <div className="rounded-3xl border border-emerald-500/20 bg-slate-900/70 p-8 shadow-2xl shadow-emerald-950/20 backdrop-blur">
                <div className="flex items-center gap-4">
                  <div className="h-12 w-12 animate-spin rounded-full border-4 border-emerald-400/30 border-t-emerald-400" />
                  <div>
                    <p className="text-xl font-semibold text-white">Inspecting the meal...</p>
                    <p className="text-sm text-slate-400">The AI is estimating nutrition.</p>
                  </div>
                </div>
              </div>
            )}

            {analysis && (
              <div className="space-y-6">
                <div className="rounded-3xl border border-emerald-500/20 bg-slate-900/70 p-6 shadow-2xl shadow-emerald-950/20 backdrop-blur">
                  <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                    <div>
                      <p className="text-sm uppercase tracking-[0.35em] text-emerald-300">Meal Analysis</p>
                      <h2 className="mt-2 text-2xl font-semibold text-white">Analysis Result</h2>
                      <p className="mt-3 text-sm leading-6 text-slate-300">
                        The AI model has analyzed the meal from the image and estimated its total nutritional values.
                      </p>
                      {analysis.analysis_source && (
                        <p className="mt-3 inline-flex rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-sm font-medium text-emerald-300">
                          {sourceLabels[analysis.analysis_source] || analysis.analysis_source}
                        </p>
                      )}
                    </div>
                    <div className="flex items-center justify-center rounded-full border border-emerald-400/40 bg-emerald-500/10 p-4">
                      <div
                        className="flex h-24 w-24 items-center justify-center rounded-full border-8 border-emerald-500/30"
                        style={{
                          background: `conic-gradient(#34d399 ${analysis.health_score * 10}%, rgba(255,255,255,0.08) 0)`
                        }}
                      >
                        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-slate-950 text-xl font-bold text-emerald-300">
                          {analysis.health_score}/10
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-2xl shadow-slate-950/40 backdrop-blur">
                  <div className="mb-5 flex items-center justify-between">
                    <h3 className="text-xl font-semibold text-white">Nutrition Totals</h3>
                    <p className="text-sm text-emerald-300">Estimated values</p>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    {metrics.map((metric) => (
                      <div key={metric.key} className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                        <p className="text-sm text-slate-400">{metric.label}</p>
                        <p className="mt-2 text-2xl font-semibold text-emerald-300">
                          {analysis.total?.[metric.key] ?? 0}
                          <span className="ml-1 text-sm text-slate-500">{metric.unit}</span>
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

export default App;
