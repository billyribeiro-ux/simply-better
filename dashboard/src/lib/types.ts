// Typed contract for the engine's static JSON export (engine/export.py).

// /data/summary.json
export interface Summary {
	generated_at: string;
	run_id: string;
	from: string;
	to: string;
	universe: string[];
	kpis: Kpis;
	per_symbol: SymbolBreakdown[];
	per_setup: SetupBreakdown[];
	feature_importance: FeatureImportance[];
	folds: number;
}

export interface Kpis {
	trades: number;
	win_rate: number;
	avg_win_usd: number;
	avg_loss_usd: number;
	avg_win_r: number;
	avg_loss_r: number;
	expectancy_r: number;
	profit_factor: number;
	net_pnl_usd: number;
	max_dd_pct: number;
	sharpe: number;
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
	setup: 'HOD_FADE' | 'LOD_RECLAIM';
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
	stop_atr: number;
	target_atr: number;
	p_win: number;
	expectancy_r: number;
	samples: number;
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
