<script lang="ts">
	import type { SymbolBreakdown } from '$lib/types';
	import { fmtUsd } from '$lib/format';

	let {
		universe,
		perSymbol
	}: { universe: string[]; perSymbol: SymbolBreakdown[] } = $props();

	const bySymbol = $derived(new Map(perSymbol.map((r) => [r.symbol, r])));
</script>

<nav class="rail" aria-label="Net profit and loss by symbol">
	<ul>
		{#each universe as symbol (symbol)}
			{@const row = bySymbol.get(symbol)}
			<li>
				<span class="ticker num">{symbol}</span>
				{#if row}
					<span
						class="pnl num"
						class:gain={row.net_pnl_usd > 0}
						class:loss={row.net_pnl_usd < 0}>{fmtUsd(row.net_pnl_usd, 0, true)}</span
					>
					<span class="sub num">{row.trades} trades · {row.win_rate.toFixed(0)}% win</span>
				{:else}
					<span class="pnl num idle">—</span>
					<span class="sub num">no trades</span>
				{/if}
			</li>
		{/each}
	</ul>
</nav>

<style>
	.rail {
		border-top: 1px solid var(--hairline);
		border-bottom: 1px solid var(--hairline);
		overflow-x: auto;
	}

	ul {
		display: flex;
		list-style: none;
		margin: 0;
		padding: 0;
		min-width: max-content;
	}

	li {
		flex: 1 0 auto;
		min-width: 128px;
		padding: 16px 20px 14px;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	li + li {
		border-left: 1px solid var(--hairline);
	}

	.ticker {
		font-size: 11px;
		font-weight: 500;
		letter-spacing: 0.12em;
		color: var(--graphite);
	}

	.pnl {
		font-size: clamp(19px, 1.9vw, 24px);
		font-weight: 500;
		line-height: 1.15;
	}

	.idle {
		color: var(--graphite);
	}

	.sub {
		font-size: 11px;
		color: var(--graphite);
	}
</style>
