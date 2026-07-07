<script lang="ts">
	import type { Kpis } from '$lib/types';
	import { fmtNum, fmtPct, fmtR, fmtUsd } from '$lib/format';

	let { kpis }: { kpis: Kpis } = $props();

	const edgeUnproven = $derived(kpis.deflated_sharpe < 0.95);
</script>

<section aria-label="Key performance indicators">
	<ul class="strip">
		<li>
			<span class="eyebrow">Net P&amp;L</span>
			<span
				class="value num"
				class:gain={kpis.net_pnl_usd > 0}
				class:loss={kpis.net_pnl_usd < 0}>{fmtUsd(kpis.net_pnl_usd, 0, true)}</span
			>
			<span class="sub num">{kpis.trades} trades</span>
		</li>
		<li>
			<span class="eyebrow">Win rate</span>
			<span class="value num">{fmtPct(kpis.win_rate)}</span>
			<span class="sub num">of taken trades</span>
		</li>
		<li>
			<span class="eyebrow">Avg win / loss</span>
			<span class="value num"
				><span class="gain">{fmtUsd(kpis.avg_win_usd, 0, true)}</span><span class="sep"> / </span><span
					class="loss">{fmtUsd(kpis.avg_loss_usd, 0)}</span
				></span
			>
			<span class="sub num">{fmtR(kpis.avg_win_r)} / {fmtR(kpis.avg_loss_r)}</span>
		</li>
		<li>
			<span class="eyebrow">Profit factor</span>
			<span class="value num">{fmtNum(kpis.profit_factor, 2)}</span>
			<span class="sub num">gross win ÷ gross loss</span>
		</li>
		<li>
			<span class="eyebrow">Expectancy</span>
			<span
				class="value num"
				class:gain={kpis.expectancy_r > 0}
				class:loss={kpis.expectancy_r < 0}>{fmtR(kpis.expectancy_r)}</span
			>
			<span class="sub num">
				{#if kpis.expectancy_r_ci?.length === 2}
					95% CI {fmtR(kpis.expectancy_r_ci[0])} … {fmtR(kpis.expectancy_r_ci[1])}
				{:else}
					per trade, in R
				{/if}
			</span>
		</li>
		<li>
			<span class="eyebrow">Max drawdown</span>
			<span class="value num" class:loss={kpis.max_dd_pct < 0}>{fmtPct(kpis.max_dd_pct)}</span>
			<span class="sub num">peak to trough</span>
		</li>
	</ul>
	<div class="stats num">
		<span>
			Sharpe {fmtNum(kpis.sharpe, 2)}{#if kpis.sharpe_ci?.length === 2}&nbsp;[{fmtNum(
					kpis.sharpe_ci[0],
					2
				)} … {fmtNum(kpis.sharpe_ci[1], 2)}]{/if}
		</span>
		<span>PSR {fmtNum(kpis.psr, 3)}</span>
		<span>DSR {fmtNum(kpis.deflated_sharpe, 3)}</span>
		{#if edgeUnproven}
			<span class="badge">Edge not proven at {kpis.n_trials_deflation} trials</span>
		{/if}
	</div>
</section>

<style>
	.strip {
		display: grid;
		grid-template-columns: repeat(6, 1fr);
		list-style: none;
		margin: 0;
		padding: 0;
		border-bottom: 1px solid var(--hairline);
	}

	li {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 20px 20px 18px 0;
	}

	li + li {
		border-left: 1px solid var(--hairline);
		padding-left: 20px;
	}

	.value {
		font-size: clamp(17px, 1.7vw, 22px);
		font-weight: 500;
		line-height: 1.2;
		white-space: nowrap;
	}

	.sep {
		color: var(--graphite);
	}

	.sub {
		font-size: 11px;
		color: var(--graphite);
	}

	.stats {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 20px;
		padding-block: 12px;
		font-size: 12px;
		color: var(--graphite);
	}

	.badge {
		color: var(--ink);
		background: var(--wash);
		border: 1px solid var(--hairline);
		padding: 3px 10px;
		font-size: 11px;
		letter-spacing: 0.04em;
	}

	@media (max-width: 900px) {
		.strip {
			grid-template-columns: repeat(2, 1fr);
		}

		li {
			border-top: 1px solid var(--hairline);
			padding-left: 0;
		}

		li:nth-child(-n + 2) {
			border-top: none;
		}

		li + li {
			border-left: none;
		}

		li:nth-child(2n) {
			border-left: 1px solid var(--hairline);
			padding-left: 20px;
		}
	}
</style>
