'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';

interface LocalProjectReport {
  project_name: string;
  analysis_timestamp: string;
  total_files: number;
  total_lines_of_code: number;
  total_size_bytes: number;
  languages: Record<string, number>;
  summary?: string;
  file_tree?: string;
}

export default function LocalProjectReportPage() {
  const params = useParams();
  const [report, setReport] = useState<LocalProjectReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        const response = await fetch(`/api/local_project/report/${params.id}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch report: ${response.statusText}`);
        }
        const data = await response.json();
        setReport(data);
      } catch (e) {
        console.error('Error fetching report:', e);
        setError(e instanceof Error ? e.message : 'Failed to load report');
      } finally {
        setLoading(false);
      }
    };

    if (params.id) {
      fetchReport();
    }
  }, [params.id]);

  if (loading) {
    return <div className="p-4">Loading project report...</div>;
  }

  if (error) {
    return <div className="p-4 text-red-500">Error: {error}</div>;
  }

  if (!report) {
    return <div className="p-4">No report found</div>;
  }

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">{report.project_name}</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-[var(--card-bg)] p-6 rounded-lg border border-[var(--border-color)]">
          <h2 className="text-xl font-semibold mb-4">Project Statistics</h2>
          <dl className="space-y-2">
            <div className="flex justify-between">
              <dt className="text-[var(--muted)]">Total Files:</dt>
              <dd>{report.total_files.toLocaleString()}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-[var(--muted)]">Lines of Code:</dt>
              <dd>{report.total_lines_of_code.toLocaleString()}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-[var(--muted)]">Total Size:</dt>
              <dd>{(report.total_size_bytes / 1024).toFixed(2)} KB</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-[var(--muted)]">Analyzed On:</dt>
              <dd>{new Date(report.analysis_timestamp).toLocaleString()}</dd>
            </div>
          </dl>
        </div>

        <div className="bg-[var(--card-bg)] p-6 rounded-lg border border-[var(--border-color)]">
          <h2 className="text-xl font-semibold mb-4">Language Distribution</h2>
          <div className="space-y-2">
            {Object.entries(report.languages)
              .sort(([, a], [, b]) => b - a)
              .map(([language, count]) => (
                <div key={language} className="flex justify-between items-center">
                  <span className="text-[var(--muted)]">{language}</span>
                  <span className="font-mono">{count} files</span>
                </div>
              ))}
          </div>
        </div>
      </div>

      {report.summary && (
        <div className="bg-[var(--card-bg)] p-6 rounded-lg border border-[var(--border-color)] mb-8">
          <h2 className="text-xl font-semibold mb-4">Project Summary</h2>
          <div className="prose prose-sm max-w-none">
            {report.summary.split('\n').map((paragraph, index) => (
              <p key={index} className="mb-4">{paragraph}</p>
            ))}
          </div>
        </div>
      )}

      {report.file_tree && (
        <div className="bg-[var(--card-bg)] p-6 rounded-lg border border-[var(--border-color)]">
          <h2 className="text-xl font-semibold mb-4">File Structure</h2>
          <pre className="overflow-x-auto text-sm font-mono whitespace-pre">
            {report.file_tree}
          </pre>
        </div>
      )}
    </div>
  );
}
