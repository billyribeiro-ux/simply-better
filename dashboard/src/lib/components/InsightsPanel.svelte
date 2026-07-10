<script lang="ts">
	import type { InsightsFile } from '$lib/types';

	let { insights }: { insights: InsightsFile } = $props();

	let symbolFilter = $state('ALL');
	const symbols = $derived(['ALL', ...new Set(insights.facts.map((f) => f.symbol))].filter(
		(s, i, a) => a.indexOf(s) === i
	));
	const shown = $derived(
		insights.facts.filter((f) => symbolFilter === 'ALL' || f.symbol === symbolFilter)
	);
</script>

<section class="shell">
	<h2 class="eyebrow head">
		Machine-discovered market facts · {insights.facts_confirmed} confirmed of
		{insights.hypotheses_tested} hypotheses
	</h2>
	<p class="method num">{insights.method}</p>
	<div class="chips">
		{#each symbols as s (s)}
			<button
				class="chip num"
				class:active={symbolFilter === s}
				aria-pressed={symbolFilter === s}
				onclick={() => (symbolFilter = s)}>{s.toLowerCase()}</button
			>
		{/each}
	</div>
	<ul class="facts">
		{#each shown.slice(0, 40) as f (f.statement)}
			<li>
				<span class="stmt">{f.statement}</span>
				<span class="num proof">
					discovered {f.discovered_pct}% (n={f.discovered_n}) · confirmed on unseen data
					{f.confirmed_pct}% (n={f.confirmed_n})
				</span>
			</li>
		{/each}
	</ul>
</section>

<style>
	.head {
		margin-bottom: 6px;
	}
	.method {
		color: var(--graphite, #666);
		font-size: 12px;
		margin: 0 0 10px;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-bottom: 12px;
	}
	.facts {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.facts li {
		border-left: 2px solid var(--rule, #ddd);
		padding-left: 12px;
	}
	.stmt {
		display: block;
	}
	.proof {
		display: block;
		font-size: 12px;
		color: var(--graphite, #666);
	}
</style>
