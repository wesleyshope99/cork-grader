// Daily log export. iOS Safari's handling of blob downloads was the single
// biggest export headache in Phase 1 (a plain <a download> rendered the raw
// file bytes on screen instead of saving), so this tries, in order:
//   1. navigator.share with a real File (the modern, reliable iOS path --
//      brings up the share sheet with "Save to Files")
//   2. a normal <a download> blob-URL click (works fine on desktop/Android)
//   3. a plain-text/CSV fallback rendered in a selectable <textarea>
//
// Callers should always offer the "view as text" fallback (3) as a visible
// button too, not just as a last resort, in case (1)/(2) silently misbehave
// on a given OS/browser combination.

const XLSX = window.XLSX;

function rowsForEntries(entries) {
  return entries.map((e) => ({
    Date: new Date(e.timestamp).toLocaleDateString(),
    Time: new Date(e.timestamp).toLocaleTimeString(),
    Grade: e.grade,
    Confidence: e.confidence != null ? Math.round(e.confidence * 100) / 100 : '',
    'Diameter (mm)': e.diameterMm ?? '',
  }));
}

export function buildXlsxBlob(entries) {
  const rows = rowsForEntries(entries);
  const ws = XLSX.utils.json_to_sheet(rows);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Log');
  const arrayBuffer = XLSX.write(wb, { type: 'array', bookType: 'xlsx' });
  return new Blob([arrayBuffer], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  });
}

export function buildCsvText(entries) {
  const rows = rowsForEntries(entries);
  if (rows.length === 0) return '';
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(',')];
  for (const row of rows) {
    lines.push(headers.map((h) => String(row[h]).replace(/,/g, ';')).join(','));
  }
  return lines.join('\n');
}

/**
 * Try to save/share the given blob as a file. Returns a string describing
 * which path succeeded: 'share' | 'download' | 'failed'.
 */
export async function saveOrShareBlob(blob, filename) {
  if (navigator.canShare && navigator.share) {
    try {
      const file = new File([blob], filename, { type: blob.type });
      if (navigator.canShare({ files: [file] })) {
        await navigator.share({ files: [file], title: filename });
        return 'share';
      }
    } catch (err) {
      // AbortError = user cancelled the share sheet; treat as handled, not a failure.
      if (err && err.name === 'AbortError') return 'share';
      // otherwise fall through to the download attempt
    }
  }

  try {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    return 'download';
  } catch (err) {
    return 'failed';
  }
}

export async function exportEntries(entries, filenamePrefix) {
  const filename = `${filenamePrefix}.xlsx`;
  const blob = buildXlsxBlob(entries);
  const result = await saveOrShareBlob(blob, filename);
  return { result, csvFallback: buildCsvText(entries), filename };
}
