<script lang="ts">
	import type { AnchorRecovery } from '$lib/types';
	import { fmtNum } from '$lib/format';

	let { rows }: { rows: AnchorRecovery[] } = $props();

	const ANCHOR_LABEL: Record<string, string> = {
		vwap: 'VWAP',
		day_open: 'day open',
		prior_close: 'prior close',
		pd_poc: 'PD POC',
		dev_poc: 'dev POC'
	};
</script>

<div class="panel">
	<h2 class="eyebrow">Reversion target evidence</h2>
	{#if rows.length}
		<div class="scroll">
			<table>
				<thead>
					<tr>
						<th scope="col">setup</th>
						<th scope="col">anchor</th>
						<th scope="col" class="r">median recovery</th>
						<th scope="col" class="r">p75</th>
						<th scope="col" class="r">n</th>
					</tr>
				</thead>
				<tbody class="num">
					{#each rows as row (row.setup + row.anchor)}
						<tr>
							<td>{row.setup}</td>
							<td>{ANCHOR_LABEL[row.anchor] ?? row.anchor}</td>
							<td class="r">{fmtNum(row.median, 3)}</td>
							<td class="r">{fmtNum(row.p75, 3)}</td>
							<td class="r">{row.samples}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="note num">fraction of the runway to each anchor that trades actually traversed</p>
	{:else}
		<p class="empty num">No positive-room trades to measure yet.</p>
	{/if}
</div>

<style>
	.panel {
		display: flex;
		flex-direction: column;
		gap: 12px;
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
		padding: 6px 12px 6px 0;
		border-bottom: 1px solid var(--hairline);
		white-space: nowrap;
	}

	.r {
		text-align: right;
	}

	tbody tr:hover {
		background: var(--wash);
	}

	.note,
	.empty {
		color: var(--graphite);
		font-size: 11px;
	}
</style>
