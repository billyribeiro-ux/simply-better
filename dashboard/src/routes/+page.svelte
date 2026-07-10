<script lang="ts">
	import type {
		Attribution,
		Equity,
		GeometryFile,
		InsightsFile,
		LiveFile,
		PaperFile,
		SignalsFile,
		Summary
	} from '$lib/types';
	import InsightsPanel from '$lib/components/InsightsPanel.svelte';
	import AnchorRecoveryPanel from '$lib/components/AnchorRecoveryPanel.svelte';
	import AttributionPanel from '$lib/components/AttributionPanel.svelte';
	import DiagnosticsPanel from '$lib/components/DiagnosticsPanel.svelte';
	import EquityCurve from '$lib/components/EquityCurve.svelte';
	import GeometryTable from '$lib/components/GeometryTable.svelte';
	import KpiStrip from '$lib/components/KpiStrip.svelte';
	import LiveSignalsPanel from '$lib/components/LiveSignalsPanel.svelte';
	import Masthead from '$lib/components/Masthead.svelte';
	import PaperTrackPanel from '$lib/components/PaperTrackPanel.svelte';
	import SignalsTable from '$lib/components/SignalsTable.svelte';
	import TickerRail from '$lib/components/TickerRail.svelte';

	interface RunData {
		summary: Summary;
		equity: Equity;
		signals: SignalsFile;
		geometry: GeometryFile;
		attribution: Attribution;
	}

	type View =
		| { status: 'loading' }
		| { status: 'error'; message: string }
		| { status: 'ready'; data: RunData };

	let view = $state<View>({ status: 'loading' });
	let liveFile = $state<LiveFile | null>(null);
	let paperFile = $state<PaperFile | null>(null);
	let insightsFile = $state<InsightsFile | null>(null);

	async function fetchJson<T>(path: string): Promise<T> {
		const res = await fetch(path);
		if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
		return (await res.json()) as T;
	}

	$effect(() => {
		let cancelled = false;
		(async () => {
			try {
				const [summary, equity, signals, geometry, attribution] = await Promise.all([
					fetchJson<Summary>('/data/summary.json'),
					fetchJson<Equity>('/data/equity.json'),
					fetchJson<SignalsFile>('/data/signals.json'),
					fetchJson<GeometryFile>('/data/geometry.json'),
					fetchJson<Attribution>('/data/attribution.json')
				]);
				if (!cancelled) {
					view = { status: 'ready', data: { summary, equity, signals, geometry, attribution } };
				}
			} catch (err) {
				if (!cancelled) {
					view = { status: 'error', message: err instanceof Error ? err.message : String(err) };
				}
			}
			// live feed and paper record are optional — absent until first runs
			try {
				const lf = await fetchJson<LiveFile>('/data/live.json');
				if (!cancelled) liveFile = lf;
			} catch {
				if (!cancelled) liveFile = null;
			}
			try {
				const pf = await fetchJson<PaperFile>('/data/paper.json');
				if (!cancelled) paperFile = pf;
			} catch {
				if (!cancelled) paperFile = null;
			}
			try {
				const inf = await fetchJson<InsightsFile>('/data/market_insights.json');
				if (!cancelled) insightsFile = inf;
			} catch {
				if (!cancelled) insightsFile = null;
			}
		})();
		return () => {
			cancelled = true;
		};
	});
</script>

<main class="page-fade">
	{#if view.status === 'loading'}
		<div class="state shell">
			<p class="eyebrow">Market intelligence engine</p>
			<p class="num msg">Loading run data…</p>
		</div>
	{:else if view.status === 'error'}
		<div class="state shell">
			<p class="eyebrow">Market intelligence engine</p>
			<p class="msg">
				No run data found. Run <code>mie run --from 2025-01-02 --to 2026-07-03</code> then refresh.
			</p>
			<p class="detail num">{view.message}</p>
		</div>
	{:else}
		{@const d = view.data}
		<Masthead summary={d.summary} />

		{#if d.summary.run_id === 'SAMPLE'}
			<p class="sample num shell">Sample data — run the engine to replace.</p>
		{/if}

		<TickerRail universe={d.summary.universe} perSymbol={d.summary.per_symbol} />

		{#if liveFile}
			<LiveSignalsPanel live={liveFile} />
		{/if}

		{#if paperFile}
			<PaperTrackPanel paper={paperFile} />
		{/if}

		{#if insightsFile && insightsFile.facts.length}
			<InsightsPanel insights={insightsFile} />
		{/if}

		<div class="shell">
			<KpiStrip kpis={d.summary.kpis} />
		</div>

		<section class="shell">
			<h2 class="eyebrow head">Equity · {d.summary.folds} walk-forward folds</h2>
			<EquityCurve equity={d.equity} />
		</section>

		<section class="shell band">
			<div class="col">
				<AttributionPanel attribution={d.attribution} />
				<AnchorRecoveryPanel rows={d.summary.anchor_recovery ?? []} />
			</div>
			<div class="col">
				<DiagnosticsPanel
					diagnostics={d.summary.diagnostics ?? { folds: [], auc_mean_test: null }}
				/>
				<GeometryTable rows={d.geometry.rows} featureImportance={d.summary.feature_importance} />
			</div>
		</section>

		<section class="shell">
			<h2 class="eyebrow head">Signals</h2>
			<SignalsTable signals={d.signals.signals} />
		</section>

		<footer class="shell num">
			run {d.summary.run_id} · {d.summary.from} → {d.summary.to} · every decision out-of-sample
		</footer>
	{/if}
</main>

<style>
	main {
		padding-bottom: 64px;
	}

	.state {
		padding-block: 96px;
		display: flex;
		flex-direction: column;
		gap: 14px;
	}

	.msg {
		font-size: 15px;
		max-width: 56ch;
	}

	.detail {
		font-size: 12px;
		color: var(--graphite);
	}

	.sample {
		border-bottom: 1px solid var(--hairline);
		font-size: 11.5px;
		color: var(--graphite);
		padding-block: 8px;
		margin: 0;
	}

	section {
		margin-top: 40px;
	}

	.head {
		margin-bottom: 14px;
	}

	.band {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 36px;
		align-items: start;
	}

	.col {
		display: flex;
		flex-direction: column;
		gap: 36px;
		min-width: 0;
	}

	.band > :global(* + *) {
		border-left: 1px solid var(--hairline);
		padding-left: 36px;
	}

	footer {
		margin-top: 48px;
		padding-top: 14px;
		border-top: 1px solid var(--hairline);
		font-size: 11px;
		color: var(--graphite);
	}

	@media (max-width: 960px) {
		.band {
			grid-template-columns: 1fr;
			gap: 40px;
		}

		.band > :global(* + *) {
			border-left: none;
			padding-left: 0;
			border-top: 1px solid var(--hairline);
			padding-top: 32px;
		}
	}
</style>
