<script lang="ts">
	import type { Attribution } from '$lib/types';
	import { fmtNum, fmtPct, fmtR } from '$lib/format';

	let { attribution }: { attribution: Attribution } = $props();

	const maxDelta = $derived.by(() => {
		const all = attribution.clusters.flatMap((c) => c.drivers.map((d) => Math.abs(d.delta)));
		return all.length ? Math.max(...all, 0.01) : 1;
	});
</script>

<div class="panel">
	<h2 class="eyebrow">Why it loses</h2>

	{#if attribution.clusters.length}
		<div class="clusters">
			{#each attribution.clusters as cluster (cluster.label)}
				<article>
					<h3 class="num">
						{cluster.label}
						<span class="meta">{cluster.size} trades · avg {fmtR(cluster.avg_pnl_r)}</span>
					</h3>
					<ul class="drivers num">
						{#each cluster.drivers as driver (driver.feature)}
							<li>
								<span class="feat">{driver.feature}</span>
								<span class="track" class:neg={driver.delta < 0}>
									<span
										class="bar"
										style:width="{(Math.abs(driver.delta) / maxDelta) * 50}%"
									></span>
								</span>
								<span class="delta">{fmtNum(driver.delta, 2)}σ</span>
								<span class="means">loss μ {fmtNum(driver.loss_mean, 2)} · win μ {fmtNum(
										driver.win_mean,
										2
									)}</span>
							</li>
						{/each}
					</ul>
				</article>
			{/each}
		</div>
	{:else}
		<p class="empty num">Not enough losing trades to cluster yet.</p>
	{/if}

	<h2 class="eyebrow rules-head">Active adjustment rules</h2>
	{#if attribution.rules.length}
		<ol class="rules">
			{#each attribution.rules as rule (rule.text + rule.learned_after_fold)}
				<li>
					<code class="cond">{rule.text}</code>
					<span class="meta num"
						>loss {fmtPct(rule.loss_rate * 100, 0)} · n {rule.support} · bump +{fmtNum(
							rule.bump,
							2
						)} · learned after {rule.learned_after_fold}</span
					>
				</li>
			{/each}
		</ol>
	{:else}
		<p class="empty num">No adjustment rules learned yet.</p>
	{/if}
</div>

<style>
	.panel {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}

	.clusters {
		display: flex;
		flex-direction: column;
		gap: 18px;
	}

	article h3 {
		font-family: var(--font-mono);
		font-size: 12.5px;
		font-weight: 600;
		display: flex;
		justify-content: space-between;
		gap: 12px;
		flex-wrap: wrap;
		padding-bottom: 6px;
		border-bottom: 1px solid var(--hairline);
	}

	.meta {
		font-weight: 400;
		font-size: 11.5px;
		color: var(--graphite);
	}

	.drivers {
		list-style: none;
		margin: 8px 0 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 6px;
		font-size: 11.5px;
	}

	.drivers li {
		display: grid;
		grid-template-columns: minmax(110px, 150px) 1fr 52px minmax(130px, auto);
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
		position: relative;
		height: 6px;
		border-bottom: 1px solid var(--hairline);
	}

	.track::before {
		content: '';
		position: absolute;
		left: 50%;
		top: -2px;
		bottom: -2px;
		width: 1px;
		background: var(--hairline);
	}

	.bar {
		display: block;
		position: absolute;
		bottom: 0;
		left: 50%;
		height: 6px;
		background: var(--ink);
	}

	.track.neg .bar {
		left: auto;
		right: 50%;
		background: var(--graphite);
	}

	.delta {
		text-align: right;
	}

	.means {
		color: var(--graphite);
		font-size: 11px;
		white-space: nowrap;
	}

	.rules-head {
		margin-top: 14px;
	}

	.rules {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
	}

	.rules li {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 10px 0;
		border-bottom: 1px solid var(--hairline);
	}

	.rules li:first-child {
		border-top: 1px solid var(--hairline);
	}

	.cond {
		font-size: 12px;
		width: fit-content;
	}

	.empty {
		color: var(--graphite);
		font-size: 12.5px;
	}

	@media (max-width: 560px) {
		.drivers li {
			grid-template-columns: minmax(100px, 130px) 1fr 48px;
		}

		.means {
			grid-column: 1 / -1;
			padding-left: 0;
		}
	}
</style>
