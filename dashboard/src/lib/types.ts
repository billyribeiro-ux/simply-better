// Typed contract for the engine's static JSON export (engine/export.py).

// /data/summary.json
export interface Summary {
	generated_at: string;
	run_id: string;
	run_type?: string | null;
	from: string;
	to: string;
	universe: string[];
	kpis: Kpis;
	per_symbol: SymbolBreakdown[];
	per_setup: SetupBreakdown[];
	feature_importance: FeatureImportance[];
	folds: number;
	diagnostics: Diagnostics;
	anchor_recovery: AnchorRecovery[];
}

export interface Diagnostics {
	folds: FoldDiagnostic[];
	auc_mean_test: number | null;
}

export interface FoldDiagnostic {
	fold: string;
	auc_test: number | null;
	auc_val: number | null;
	threshold: number;
	n_test: number;
	prob_hist: number[];
}

export interface AnchorRecovery {
	setup: string;
	anchor: string;
	median: number;
	p75: number;
	samples: number;
}

export interface Kpis {
	trades: number;
	win_rate: number;
	avg_win_usd: number;
	avg_loss_usd: number;
	avg_win_r: number;
	avg_loss_r: number;
	expectancy_r: number;
	expectancy_r_ci?: number[]; // 95% bootstrap CI [lo, hi]
	profit_factor: number;
	net_pnl_usd: number;
	max_dd_pct: number;
	sharpe: number;
	sharpe_ci?: number[]; // 95% block-bootstrap CI [lo, hi]
	psr: number;
	deflated_sharpe: number;
	n_trials_deflation: number;
}

export interface SymbolBreakdown {
	symbol: string;
	trades: number;
	win_rate: number;
	net_pnl_usd: number;
}

export interface SetupBreakdown {
	setup: string;
	trades: number;
	win_rate: number;
	net_pnl_usd: number;
}

export interface FeatureImportance {
	feature: string;
	gain: number;
}

// /data/equity.json — curve.length === dates.length + 1
export interface Equity {
	dates: string[];
	curve: number[];
}

// /data/signals.json
export interface SignalsFile {
	signals: Signal[];
}

export interface Signal {
	id: number;
	symbol: string;
	date: string;
	setup: string;
	side: 'SHORT' | 'LONG';
	trigger_ts: string;
	entry_ts: string;
	entry_px: number;
	stop_px: number;
	target_px: number;
	exit_ts: string;
	exit_px: number;
	exit_reason: 'target' | 'stop' | 'eod';
	prob: number;
	threshold: number;
	taken: boolean;
	outcome: 'WIN' | 'LOSS';
	net_ev_r?: number; // expected value in R net of that name's execution cost
	pnl_r: number;
	pnl_usd: number | null;
	shares: number | null;
	mae_r: number;
	mfe_r: number;
}

// /data/geometry.json
export interface GeometryFile {
	rows: GeometryRow[];
}

export interface GeometryRow {
	setup: string;
	regime: number;
	anchor: string; // "atr" = fixed ATR-multiple targets
	frac: number; // fraction of anchor distance (0 in atr mode)
	stop_atr: number;
	target_atr: number;
	p_win: number;
	expectancy_r: number;
	samples: number;
}

// /data/live.json — written by `mie live`; may be absent until the first run
export interface LiveFile {
	generated_at: string;
	session_date: string;
	as_of_et: string;
	trained_through: string;
	threshold_base: number;
	equity: number;
	signals: LiveSignal[];
	note?: string;
}

export interface LiveSignal {
	id: string;
	symbol: string;
	setup: string;
	side: 'SHORT' | 'LONG';
	trigger_ts: string;
	prob: number;
	threshold: number;
	taken: boolean;
	tradable: boolean;
	trend_veto?: boolean;
	day_type_eff?: number;
	anchor: string;
	frac: number;
	stop_atr: number;
	target_atr: number;
	entry_trigger_px: number;
	stop_px: number;
	target_px: number;
	shares: number;
	status: 'awaiting' | 'confirmed' | 'expired';
	entry_ts: string | null;
	entry_px: number | null;
	max_room_atr: number;
}

// /data/paper.json — the forward paper-trading record; absent until the
// first `mie paper` resolution
export interface PaperFile {
	generated_at: string;
	equity_start: number;
	kpis: {
		trades: number;
		open: number;
		win_rate: number | null;
		net_pnl_usd: number;
		expectancy_r: number | null;
	};
	equity: Equity;
	trades: PaperTrade[];
}

export interface PaperTrade {
	session_date: string;
	signal_id: string;
	symbol: string;
	setup: string;
	side: 'SHORT' | 'LONG';
	trigger_ts: string;
	entry_ts: string;
	entry_px: number;
	shares: number;
	stop_px: number;
	target_px: number;
	exit_ts: string;
	exit_px: number;
	exit_reason: 'target' | 'stop' | 'eod' | 'open';
	resolved: boolean;
	outcome: 'WIN' | 'LOSS';
	pnl_r: number;
	pnl_usd: number;
	prob: number;
	threshold: number;
	trained_through: string;
	resolved_at: string;
}

// /data/attribution.json
export interface Attribution {
	rules: AdjustmentRule[];
	clusters: LossCluster[];
}

export interface AdjustmentRule {
	text: string;
	loss_rate: number;
	support: number;
	bump: number;
	learned_after_fold: string;
}

export interface LossCluster {
	label: string;
	size: number;
	avg_pnl_r: number;
	drivers: ClusterDriver[];
}

export interface ClusterDriver {
	feature: string;
	delta: number;
	loss_mean: number;
	win_mean: number;
}
