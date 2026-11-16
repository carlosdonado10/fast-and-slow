/** Report order. Feature ids match report.features (from the spec's feature map). */
export const SECTIONS = [
  { id: 'answer', fid: 'F1', title: 'The question' },
  { id: 'scorecard', fid: 'F2', title: 'Hypothesis scorecard' },
  { id: 'desk', fid: 'F3', title: 'The desk' },
  { id: 'learning', fid: 'F4', title: 'Learning curves' },
  { id: 'efficiency', fid: 'F4b', title: 'Sample efficiency' },
  { id: 'horizon', fid: 'F5', title: 'Horizon dial' },
  { id: 'handoff', fid: 'F6', title: 'Hand-off' },
  { id: 'replays', fid: 'F7', title: 'Decision replays' },
  { id: 'costs', fid: 'F8', title: 'Cost and latency' },
  { id: 'decisions', fid: 'F9', title: 'Decisions log' },
  { id: 'limitations', fid: 'F10', title: 'Limitations' },
] as const;
