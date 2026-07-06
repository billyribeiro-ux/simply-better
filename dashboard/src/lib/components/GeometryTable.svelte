<script lang="ts">
	import type { FeatureImportance, GeometryRow } from '$lib/types';
	import { fmtNum, fmtR } from '$lib/format';

	let {
		rows,
		featureImportance
	}: { rows: GeometryRow[]; featureImportance: FeatureImportance[] } = $props();

	const REGIMES = ['low', 'mid', 'high'] as const;

	function regimeLabel(r: number): string {
		return r >= 0 && r < REGIMES.length ? REGIMES[r] : 'pooled';
	}

	const maxGain = $derived(
		featureImportance.length ? Math.max(...featureImportance.map((f) => f.gain)) : 1
	);
</script>

<div class="panel">
	<h2 class="eyebrow">Learned geometry</h2>
	{#if rows.length}
		<div class="scroll">
			<table>
				<thead>
					<tr>
						<th scope="col">setup</th>
						<th scope="col">vol regime</th>
						<th scope="col" class="r">stop ATR</th>
						<th scope="col" class="r">target ATR</th>
						<th scope="col" class="r">p(win)</th>
						<th scope="col" class="r">expectancy</th>
						<th scope="col" class="r">n</th>
					</tr>
				</thead>
				<tbody class="num">
					{#each rows as row (row.setup + row.regime)}
						<tr>
							<td>{row.setup}</td>
							<td>{regimeLabel(row.regime)}</td>
							<td class="r">{fmtNum(row.stop_atr, 2)}</td>
							<td class="r">{fmtNum(row.target_atr, 2)}</td>
							<td class="r">{fmtNum(row.p_win, 3)}</td>
							<td
								class="r"
								class:gain={row.expectancy_r > 0}
								class:loss={row.expectancy_r < 0}>{fmtR(row.expectancy_r)}</td
							>
							<td class="r">{row.samples}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else}
		<p class="empty num">No geometry learned in this run.</p>
	{/if}

	<h2 class="eyebrow imp-head">Feature importance</h2>
	{#if featureImportance.length}
		<ul class="importance num">
			{#each featureImportance as fi (fi.feature)}
				<li>
					<span class="feat">{fi.feature}</span>
					<span class="track"><span class="bar" style:width="{(fi.gain / maxGain) * 100}%"
						></span></span>
					<span class="val">{fmtNum(fi.gain, 3)}</span>
				</li>
			{/each}
		</ul>
	{:else}
		<p class="empty num">No model folds in this run.</p>
	{/if}
</div>

<style>
	.panel {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}

	.scroll {
		overflow-x: auto;
	}

	table {
		width: 100%;
		font-size: 12.5px;
	}

	th {
		font-family: var(--font-mono);
		font-size: 10.5px;
		font-weight: 500;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--graphite);
		text-align: left;
		padding: 8px 12px 8px 0;
		border-bottom: 1px solid var(--hairline);
		white-space: nowrap;
	}

	td {
		padding: 7px 12px 7px 0;
		border-bottom: 1px solid var(--hairline);
		white-space: nowrap;
	}

	.r {
		text-align: right;
	}

	tbody tr:hover {
		background: var(--wash);
	}

	.imp-head {
		margin-top: 14px;
	}

	.importance {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 6px;
		font-size: 11.5px;
	}

	.importance li {
		display: grid;
		grid-template-columns: minmax(120px, 160px) 1fr 48px;
		align-items: center;
		gap: 10px;
	}

	.feat {
		color: var(--graphite);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.track {
		display: block;
		height: 6px;
		border-bottom: 1px solid var(--hairline);
		position: relative;
	}

	.bar {
		display: block;
		position: absolute;
		bottom: 0;
		left: 0;
		height: 6px;
		background: var(--ink);
	}

	.val {
		text-align: right;
		color: var(--graphite);
	}

	.empty {
		color: var(--graphite);
		font-size: 12.5px;
	}
</style>
