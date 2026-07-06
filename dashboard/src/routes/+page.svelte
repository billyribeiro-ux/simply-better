<script lang="ts">
	import type {
		Attribution,
		Equity,
		GeometryFile,
		SignalsFile,
		Summary
	} from '$lib/types';
	import AttributionPanel from '$lib/components/AttributionPanel.svelte';
	import EquityCurve from '$lib/components/EquityCurve.svelte';
	import GeometryTable from '$lib/components/GeometryTable.svelte';
	import KpiStrip from '$lib/components/KpiStrip.svelte';
	import Masthead from '$lib/components/Masthead.svelte';
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

		<div class="shell">
			<KpiStrip kpis={d.summary.kpis} />
		</div>

		<section class="shell">
			<h2 class="eyebrow head">Equity · {d.summary.folds} walk-forward folds</h2>
			<EquityCurve equity={d.equity} />
		</section>

		<section class="shell band">
			<AttributionPanel attribution={d.attribution} />
			<GeometryTable rows={d.geometry.rows} featureImportance={d.summary.feature_importance} />
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
