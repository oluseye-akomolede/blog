---
layout: post
title: "How AI Player Plans a Job and Then Plays It"
date: 2026-11-18
---

A Minecraft server is a shared world where every player does their own work. AI Player puts bots on that server which are players in the engine's own sense of the word, and lets the people in the world hand them jobs in plain English through chat. What makes it work is a split: the model plans, and ordinary code plays.

## What AI Player is

AI Player is a Minecraft mod plus a Python agent. The mod runs inside a NeoForge 1.21.1 server and spawns each bot as a real server-side player entity, a subclass of the game's own `ServerPlayer` ([BotServerPlayer.java:10](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/bot/BotServerPlayer.java#L10)). A bot is not a mob with a script attached, and it is not a network client pretending to be a player. It is the same kind of object the server uses for a person who has logged in.

People give the bots orders in plain English in in-game chat, for example "mine 20 iron and smelt it". The bots plan the job and carry it out in the world next to them, while the people keep playing.

Language models do the thinking at the top of the system: they break an order into steps and handle the cases the routine path does not cover. Ordinary Java code does the moment to moment play, inside the server's own game tick ([AIPlayerMod.java:177](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/AIPlayerMod.java#L177)).

Every claim about the code in this post is a claim about commit `6d0c04f` of the public repository, where the paths and line numbers I cite can be checked.

## What it is for

The purpose is a modded survival server where a player can hand long, tedious jobs to a crew of helpers: mining, smelting, crafting, searching, storing, fighting. The player keeps giving orders in chat and can see what the crew is doing on a web dashboard.

The bots are meant to work as part of a group of people: the people write the orders, they answer the questions the system asks them, and they decide whether a harder question leaves the local model at all. The system is built to keep that arrangement intact even when the people are away from the keyboard.

## What it is good at

There are no performance benchmarks in the repository. The numbers in this section are configured limits and counts of code, not measured speeds or success rates. They describe what the system is built to guarantee, not what a run has demonstrated.

**A wide action vocabulary.** The mod has 37 directive types, from mining, crafting, smelting and combat to container storage, vehicles and teleport ([DirectiveType.java:3](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/DirectiveType.java#L3)). A planner composes those types into jobs.

**The two halves run at different speeds, on purpose.** The mod plays every server tick ([BotPlayer.java:502](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/bot/BotPlayer.java#L502)). The agent loop is set to check in every two seconds by default ([config.py:89](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/config.py#L89)). The agent is a separate process from the server, so a slow model call delays the bot's next decision while the game keeps ticking.

**The fast path is bounded.** Loops carry a `max_iterations` requirement ([SkillBehavior.java:221](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/behavior/SkillBehavior.java#L221)), a skill run is capped at 32 nested skill calls and 500 executed actions ([SkillBehavior.java:38-39](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/behavior/SkillBehavior.java#L38-L39)), and the validator refuses a skill tree deeper than 64 ([SkillValidator.java:50](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/skill/SkillValidator.java#L50)). A badly written skill does not run forever.

**The skill library cannot grow without limit.** It holds 64 skills, evicts the least recently used first, and never evicts the hand-written seed skills ([SkillRegistry.java:37](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/skill/SkillRegistry.java#L37), [SkillRegistry.java:95](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/skill/SkillRegistry.java#L95)).

**A result can be checked against the world.** A skill may carry an optional `verify` rule, which turns a finished run into a failed one when the inventory does not show the expected result ([SkillSpec.java:23](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/skill/SkillSpec.java#L23)). Whether a step is done is decided in the same spirit: the system asks the world first, then looks at the last action's result, and only then asks a model for a yes or no ([criteria_eval.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/criteria_eval.py#L1)). The model does not get the first word on its own work.

**An absent player does not wedge the crew.** When the system needs the player's ruling, the question goes to the player's in-game inbox and the plan waits. A question times out (90 seconds for the planner's questions, 180 for an explicit ask-player order) and the bot falls back to what it would have done before ([plan_orchestrator.py:649](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/plan_orchestrator.py#L649), [agent.py:930](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/agent.py#L930), [escalation.py:5](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/escalation.py#L5)).

**Optional services cannot take the bots down with them.** The state compressor, the external model and the skill author are all best-effort ([nn_compressor.py:3](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/nn_compressor.py#L3), [l5.py:3](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/l5.py#L3), [skill_author.py:21](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/skill_author.py#L21)).

**A slow server cannot hang the bot-control API.** Every route goes through one bounded-wait wrapper that returns 503 instead of hanging, and the code puts the number of waiting call sites that pass through it at 47 ([BoundedHandler.java:14](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/api/BoundedHandler.java#L14), [HttpApiServer.java:96](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/api/HttpApiServer.java#L96)).

## The system, in two halves

The full system is one Java mod inside the server and one Python agent outside it, talking over HTTP. The diagram shows the parts and the direction of control.

![The whole system: player chat goes to the mod, the mod exposes an HTTP API to the Python agent, the agent observes chat and world state, matches the order to a known skill or runs the L2 orchestrator with the L3 planner, evaluates criteria and can escalate to the player for an L4 ruling and onward to the optional L5 model, then sends a directive back to the mod, which plays it every server tick and reports progress and results; the trajectory log, skill author, dashboard and task board sit on side channels]({{ site.baseurl }}/assets/ai-player-architecture/full-system.svg)

**The mod.** It spawns the bots, runs their behaviors every tick, and exposes the HTTP API the agent uses ([AIPlayerMod.java:177](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/AIPlayerMod.java#L177), [HttpApiServer.java:106](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/api/HttpApiServer.java#L106)). Each bot has a `BotBrain` that holds its current directive, picks the matching behavior, and drives it one tick at a time until it reports success, failure or cancellation; a reflex layer runs before the behavior each tick and can take over, for example to keep the bot alive ([BotBrain.java:72](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/BotBrain.java#L72)). There is one Java class per kind of work, 38 of them in `brain/behavior/`, plus low-level actions ([brain/behavior](https://github.com/oluseye-akomolede/ai-player/tree/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/behavior)).

**The skill engine.** A skill is a declarative tree rather than code: six node types, the composition nodes `sequence` and `fallback`, the control nodes `loop` and `if`, a reference to another skill, and directive leaves ([SkillNode.java:36](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/skill/SkillNode.java#L36)). The engine walks the tree on a stack of frames, starts one child behavior per leaf, and drives it tick by tick until it settles ([SkillBehavior.java:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/behavior/SkillBehavior.java#L1)).

**The agent.** It is one process with one thread per bot: an observe, think and act loop that reads chat and world state, plans, and sends directives to the mod ([agent.py:152](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/agent.py#L152), [agent.py:2584](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/agent.py#L2584)). A directive goes out as a POST to the bot's directive route and progress comes back on the same route ([api.py:428](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/api.py#L428), [api.py:43](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/api.py#L43)). Around that loop sit the skill matcher, the orchestrator, the planner, the escalation path, memory, a shared task board for several bots, and a dashboard that serves a web app from a background thread ([taskboard.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/taskboard.py#L1), [server.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/dashboard/server.py#L1)).

## Five decision tiers, L1 to L5

The code names its layers L1 to L5, from fastest and cheapest to slowest and most expensive. The last two are not automatic: they exist so the system can stop and ask.

**L1, the mod, every server tick.** It executes one directive or one skill and handles reflexes. No model is involved.

**L2, the orchestrator, deterministic.** It owns the plan and moves through its subtasks, deciding whether each one is done by checking the world first, then retrying or re-planning ([plan_orchestrator.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/plan_orchestrator.py#L1), [plan_orchestrator.py:193](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/plan_orchestrator.py#L193)). In front of it sits the skill matcher, which compares the order with a known seed skill in code, with no model call, and refuses rather than guesses when the phrase does not fit ([skill_matcher.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/skill_matcher.py#L1), [skill_matcher.py:388](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/skill_matcher.py#L388)). An order that matches a skill skips the model entirely.

**L3, the local planner model.** It makes three kinds of call: one plan per task, one execution call per subtask to turn it into directives, and a re-plan when a subtask has exhausted its retries ([l3_planner.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/l3_planner.py#L1), [l3_planner.py:398](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/l3_planner.py#L398), [l3_planner.py:888](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/l3_planner.py#L888), [l3_planner.py:983](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/l3_planner.py#L983)). It runs on local hardware.

**L4, the player.** When the system should not decide alone, for example with an ambiguous item or an explicit request to ask, the question goes to the player's in-game inbox and the plan blocks until a ruling arrives or the wait times out ([escalation.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/escalation.py#L1), [escalation.py:39](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/escalation.py#L39)).

**L5, an external model.** It is one stateless advisory call to a hosted model, made only when the player chooses it for a question and using the player's own API key ([l5.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/l5.py#L1), [l5.py:24](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/l5.py#L24)). The answer comes back to the player as a new inbox item for approval. Nothing from L5 reaches a bot unreviewed.

Off the live loop, a stronger model runs as the skill author: it mines finished plans out of the trajectory log, asks the model to generalize a two to four step routine into a parameterized skill, and registers the result through the mod's skills endpoint, which validates a skill but never runs it ([skill_author.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/skill_author.py#L1), [skill_author.py:345](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/skill_author.py#L345)).

## The model plans, the code plays

The close-up shows how one order moves through the system.

![The plan and play loop: the L3 model turns an order into a plan and turns each subtask into a directive, which the mod plays every server tick through a skill or behavior under depth, iteration and action bounds; the result and progress counters come back over the API, and the orchestrator checks the world before deciding to move on, retry or re-plan, while every planner call is written to the trajectory log; off the live loop, the skill author generalizes routines into JSON skills, which are validated (known step types, no self-reference, depth cap 64) and stored, never run at registration]({{ site.baseurl }}/assets/ai-player-architecture/plans-plays.svg)

**Plan, once per task.** L3 turns the order into a plan, and then turns each subtask into directives ([l3_planner.py:398](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/l3_planner.py#L398), [l3_planner.py:888](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/l3_planner.py#L888)). Off the live loop, the skill author writes whole reusable skills as JSON trees.

**Check before running.** The mod validates a skill before storing it: known directive types, no self-reference, a depth cap of 64 ([SkillValidator.java:24](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/skill/SkillValidator.java#L24), [SkillValidator.java:50](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/skill/SkillValidator.java#L50)). Registering a skill never runs it.

**Play, every tick.** `BotBrain.tick()` hands the directive to its behavior. For a skill directive, the engine walks the tree one action at a time and drives each leaf's behavior until it settles ([SkillBehavior.java:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/behavior/SkillBehavior.java#L1)).

**Bounded.** The loops carry `max_iterations`, with at most 32 nested skill calls and at most 500 actions per run.

**Report.** The result and the progress counters come back over the API. L2 checks the world and decides whether to move on, retry or re-plan ([criteria_eval.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/criteria_eval.py#L1), [plan_orchestrator.py:193](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/plan_orchestrator.py#L193)). Each planner call and its outcome goes to the trajectory log ([trajectory_log.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/trajectory_log.py#L1)).

The key design choice: the model is called once per plan or subtask, not once per game tick. The fast loop is plain code that can be read, validated and bounded. The game never waits on a model.

## Related work

The repository's code does not cite any of these. They are related work, not dependencies and not the basis of the design.

**Voyager** (Wang et al., 2023, "Voyager: An Open-Ended Embodied Agent with Large Language Models") is the closest idea: a language model agent in Minecraft that builds a library of reusable skills from experience. The skill author here follows the same thought. The difference is what a skill is. Voyager's skills are JavaScript programs run through Mineflayer; here they are declarative JSON trees, validated before storage and interpreted inside the server ([skill_author.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/skill_author.py#L1)).

**Mineflayer** is the common JavaScript bot library, and it connects to a server as a network client. AI Player does not use it. Its bots are player entities on the server side, which is why the mod can reach the game's own tick, inventory and world state without a client in the way.

**Behavior trees** are the sequence and fallback composition used in game AI and robotics. The skill node types `sequence` and `fallback` follow that pattern ([SkillNode.java:36](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/src/main/java/com/oluseyeakomolede/aiplayermod/brain/skill/SkillNode.java#L36)).

**Direct Preference Optimization** (Rafailov et al., 2023) is the method behind the training script for the planner, which uses the DPOTrainer from Hugging Face's TRL ([train_dpo.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/train_dpo.py#L1)).

## Where the evidence stops

There are no benchmarks in the repository, so this post claims no speed, success rate or win rate. The learning pipeline is real and complete on paper: the trajectory log records every planner call with its outcome, a deterministic reward function scores those records, a preference dataset is built from them, and a training script exists ([rl_reward.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/rl_reward.py#L1), [build_dpo_dataset.py:1](https://github.com/oluseye-akomolede/ai-player/blob/6d0c04f/agent/build_dpo_dataset.py#L1)). I am not claiming the planner was trained with it, or improved by it.

The public tests are narrower than the system. Four Java test classes with 43 methods cover the API and health layer ([src/test](https://github.com/oluseye-akomolede/ai-player/tree/6d0c04f/src/test/java/com/oluseyeakomolede/aiplayermod/api)), and 37 Python test functions across two files cover readiness and chat classification; a script-style test checks the reward function (agent/rl_reward_test.py). The planner and the skill engine have no tests in the public repository, so I am not going to call them well tested.

What the system offers instead is boundedness. Every loop, nesting depth and action count in the skill engine has a hard limit, and every API wait is bounded. The slow parts that decide are kept apart from the fast part that plays. Bounded is not the same as measured, and the limits above are limits, not results.
