<script lang="ts">
	import type { PaperFile } from '$lib/types';
	import { fmtDate, fmtNum, fmtPct, fmtR, fmtTime, fmtUsd } from '$lib/format';

	let { paper }: { paper: PaperFile } = $props();

	const k = $derived(paper.kpis);
</script>

<section class="shell" aria-label="Paper track record">
	<div class="head">
		<h2 class="eyebrow">Paper track record · forward out-of-sample</h2>
		<p class="meta num">
			{k.trades} resolved{k.open ? ` · ${k.open} open` : ''}
			{#if k.win_rate !== null}
				· win {fmtPct(k.win_rate)} · net
				<span class:gain={k.net_pnl_usd > 0} class:loss={k.net_pnl_usd < 0}
					>{fmtUsd(k.net_pnl_usd, 0, true)}</span
				>
				{#if k.expectancy_r !== null}
					· {fmtR(k.expectancy_r)}
				{/if}
			{/if}
		</p>
	</div>

	{#if paper.trades.length}
		<div class="scroll">
			<table>
				<thead>
					<tr>
						<th scope="col">date</th>
						<th scope="col">sym</th>
						<th scope="col">setup</th>
						<th scope="col">side</th>
						<th scope="col" class="r">entry</th>
						<th scope="col" class="r">stop</th>
						<th scope="col" class="r">target</th>
						<th scope="col" class="r">exit</th>
						<th scope="col">via</th>
						<th scope="col" class="r">shares</th>
						<th scope="col" class="r">R</th>
						<th scope="col" class="r">P&amp;L $</th>
					</tr>
				</thead>
				<tbody class="num">
					{#each paper.trades as t (t.session_date + t.signal_id)}
						<tr class:open-row={!t.resolved}>
							<td>{fmtDate(t.session_date)}</td>
							<td>{t.symbol}</td>
							<td class="dim">{t.setup}</td>
							<td class:loss={t.side === 'SHORT'} class:gain={t.side === 'LONG'}>{t.side}</td>
							<td class="r">{fmtTime(t.entry_ts)} <span class="dim">{fmtNum(t.entry_px, 2)}</span></td>
							<td class="r loss">{fmtNum(t.stop_px, 2)}</td>
							<td class="r gain">{fmtNum(t.target_px, 2)}</td>
							<td class="r">{fmtTime(t.exit_ts)} <span class="dim">{fmtNum(t.exit_px, 2)}</span></td>
							<td class="dim">{t.exit_reason}</td>
							<td class="r">{t.shares}</td>
							<td class="r" class:gain={t.pnl_r > 0} class:loss={t.pnl_r < 0}>{fmtR(t.pnl_r)}</td>
							<td class="r" class:gain={t.pnl_usd > 0} class:loss={t.pnl_usd < 0}>
								{fmtUsd(t.pnl_usd, 0, true)}{#if !t.resolved}<span class="dim"> (open)</span>{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="note num">
			resolved with backtest rules: first crossing of stop/target on 1-min bars, ties against
			the trade, 15:55 flat, slippage + commission charged. Only signals after the model's
			trained-through date count toward confirmation.
		</p>
	{:else}
		<p class="empty num">
			No paper trades yet. Run <code>mie live --poll 60</code> during the session (auto-resolves
			at the close) or <code>mie paper</code> after it.
		</p>
	{/if}
</section>

<style>
	section {
		margin-top: 28px;
		padding-bottom: 24px;
		border-bottom: 1px solid var(--hairline);
	}

	.head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 16px;
		flex-wrap: wrap;
		margin-bottom: 10px;
	}

	.meta {
		font-size: 11.5px;
		color: var(--graphite);
	}

	.scroll {
		overflow-x: auto;
		border-top: 1px solid var(--hairline);
	}

	table {
		width: 100%;
		font-size: 12px;
		min-width: 920px;
	}

	th {
		font-family: var(--font-mono);
		font-size: 10.5px;
		font-weight: 500;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--graphite);
		text-align: left;
		padding: 8px 10px 8px 0;
		border-bottom: 1px solid var(--hairline);
		white-space: nowrap;
	}

	th.r {
		text-align: right;
	}

	td {
		padding: 6px 10px 6px 0;
		border-bottom: 1px solid var(--hairline);
		white-space: nowrap;
	}

	td.r {
		text-align: right;
	}

	tbody tr:hover {
		background: var(--wash);
	}

	tr.open-row {
		opacity: 0.7;
	}

	.dim {
		color: var(--graphite);
	}

	.note,
	.empty {
		color: var(--graphite);
		font-size: 11px;
		margin-top: 8px;
	}
</style>
