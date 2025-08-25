import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  Legend,
  CartesianGrid,
} from "recharts";

// Minimal shadcn-like primitives (so this single-file preview still looks nice)
const Card = ({ children }) => (
  <div className="rounded-2xl shadow-md border border-gray-200 bg-white p-5">{children}</div>
);
const H = ({ children, className = "" }) => (
  <h2 className={`text-xl font-semibold tracking-tight ${className}`}>{children}</h2>
);
const Small = ({ children }) => (
  <p className="text-xs text-gray-500">{children}</p>
);
const Button = ({ children, onClick, className = "" }) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 rounded-xl shadow-sm border border-gray-200 hover:shadow transition ${className}`}
  >
    {children}
  </button>
);

const API_BASE = import.meta?.env?.VITE_API_BASE || "http://localhost:8000";

function useApi(path, { auto = true } = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(auto);
  const [error, setError] = useState(null);
  const url = `${API_BASE}${path}`;

  const fetchIt = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (auto) fetchIt();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  return { data, loading, error, refetch: fetchIt };
}

function fmtDate(d) {
  const dt = new Date(d);
  return dt.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "2-digit" });
}

export default function App() {
  const metrics = useApi("/api/metrics");
  const pred = useApi("/api/predict");
  const comments = useApi("/api/comments?limit=50");

  const merged = metrics.data?.merged || [];

  const priceSeries = useMemo(() => {
    return merged.map((r) => ({
      date: r.date,
      Close: r.Close,
      PredictedNextClose: r.predicted_next_close ?? null,
    }));
  }, [merged]);

  const sentimentSeries = useMemo(() => {
    return merged.map((r) => ({
      date: r.date,
      positive: r.positive ?? 0,
      neutral: r.neutral ?? 0,
      negative: r.negative ?? 0,
    }));
  }, [merged]);

  const last = merged.length ? merged[merged.length - 1] : null;

  return (
    <div className="min-h-screen w-full bg-gray-50 text-gray-900">
      <header className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">NVDA Sentiment + Price Predictor</h1>
            <Small>Self-updating dashboard powered by a Python (FastAPI) backend</Small>
          </div>
          <div className="flex gap-2">
            <Button onClick={() => { metrics.refetch(); pred.refetch(); comments.refetch(); }}>Refresh</Button>
            <a
              href={`${API_BASE}/docs`}
              target="_blank"
              rel="noreferrer"
              className="px-4 py-2 rounded-xl shadow-sm border border-gray-200 hover:shadow transition"
            >
              API Docs
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: charts */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <div className="flex items-center justify-between mb-4">
              <H>NVDA Close Price & Next-Day Prediction</H>
              <Small>{metrics.loading || pred.loading ? "loading…" : last ? `Last: ${fmtDate(last.date)}` : ""}</Small>
            </div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={priceSeries} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={fmtDate} minTickGap={24} />
                  <YAxis dataKey="Close" domain={["auto", "auto"]} />
                  <Tooltip labelFormatter={(v) => fmtDate(v)} />
                  <Legend />
                  <Line type="monotone" dataKey="Close" dot={false} strokeWidth={2} />
                  {/* If backend injects predicted_next_close on last row, show it */}
                  <Line type="monotone" dataKey="PredictedNextClose" dot={true} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between mb-4">
              <H>Reddit Finance Sentiment (normalized)</H>
              <Small>Positive vs Neutral vs Negative over time</Small>
            </div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={sentimentSeries} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={fmtDate} minTickGap={24} />
                  <YAxis domain={[0, 1]} />
                  <Tooltip labelFormatter={(v) => fmtDate(v)} />
                  <Legend />
                  <Area type="monotone" dataKey="positive" stackId="1" />
                  <Area type="monotone" dataKey="neutral" stackId="1" />
                  <Area type="monotone" dataKey="negative" stackId="1" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        {/* Right column: KPIs & comments */}
        <div className="space-y-6">
          <Card>
            <H className="mb-2">Snapshot</H>
            <div className="grid grid-cols-2 gap-4">
              <Kpi label="Latest Close" value={last?.Close ? `$${last.Close.toFixed(2)}` : "—"} />
              <Kpi label="Predicted Next Close" value={pred.data?.predicted_next_close ? `$${pred.data.predicted_next_close.toFixed(2)}` : "—"} />
              <Kpi label="Predicted Next Return" value={pred.data?.predicted_next_return ? `${(pred.data.predicted_next_return * 100).toFixed(2)}%` : "—"} />
              <Kpi label="# Comments (30d)" value={metrics.data?.stats?.comment_count_30d ?? "—"} />
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between mb-2">
              <H>Recent Finance Comments</H>
              <Small>Top 50 by score</Small>
            </div>
            <div className="max-h-96 overflow-auto border border-gray-100 rounded-xl">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-gray-50">
                  <tr>
                    <th className="text-left p-2">Date</th>
                    <th className="text-left p-2">Author</th>
                    <th className="text-left p-2">Sentiment</th>
                    <th className="text-left p-2">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {(comments.data?.items || []).map((c) => (
                    <tr key={c.comment_id} className="border-t">
                      <td className="p-2 whitespace-nowrap">{fmtDate(c.created_utc)}</td>
                      <td className="p-2">{c.author}</td>
                      <td className="p-2 capitalize">{c.sentiment}</td>
                      <td className="p-2">{c.score}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </main>

      <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-8">
        <Small>
          Tip: Set <code>VITE_API_BASE</code> in your frontend env (defaults to http://localhost:8000). Backend exposes
          <code> /api/metrics</code>, <code>/api/predict</code>, and <code>/api/comments</code>.
        </Small>
      </footer>
    </div>
  );
}

function Kpi({ label, value }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-gray-200 p-4"
    >
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
    </motion.div>
  );
}

