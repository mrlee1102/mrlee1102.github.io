---
layout: post
title: A Brief Survey of Deep Reinforcement Learning (2)
date: 2025-04-21 15:26 +0800
tags: [Reinforcement Learning, Deep Learning, RL Algorithm, Survey]
toc: true
comments: true
math: true
---

[Paper Link](https://arxiv.org/pdf/1708.05866)  <br  />

<em>A Brief Survey of Deep Reinforcement Learning 논문의 'Reinforcement learning algorithms - Value functions - Policy search' 까지 내용을 다루는 글</em>

## 3. REINFORCEMENT LEARNING ALGORITHMS
앞선 [내용](https://mrlee1102.github.io/2025/04/20/a-brief-survey-of-deep-reinforcement-learning-1/)에서는 RL에서 사용되는 주요 형식주의인 MDP에 대한 소개와 RL의 몇 가지 도전과제를 간략하게 살펴보았다. 이후 내용은 다양한 강화학습 알고리즘을 소개하고 구분한다. 먼저, 강화학습 문제를 푸는 데는 크게 **두 가지 주요 접근법**이 있다. **가치 함수 (Value Function) 기반**과 **정책 탐색 (Policy Search) 기반** 방법이다. 또한 이들의 **하이브리드 접근법인 행위자-비판적 (Actor-Critic) 접근법**이 있다.

### 3. A. Value Function
**가치 함수 기반 방법**은 **특정 상태에 있을 때의 가치(Expected Return, 기대 반환)를 추정**하는 방법이다.
**상태 가치 함수 (State-Value Function)** $V^{\pi}(\mathrm{s})$는 상태 $\mathrm{s}$에서 시작하여 정책 $\pi$를 따랐을 때 반환(Return)의 기대값을 의미한다:
$$ V^{\pi}(\mathrm{s})=\mathbb{E}[R\mid \mathrm{s}, \pi]$$
 
문제 해결 또는 기대보상을 최대화하는 최적 정책을 찾는 것이 강화학습의 목표인데 **최적 정책 $\pi^{*}$에는 이에 대응하는 최적 상태 가치 함수** $V^{*}(\mathrm{s})$가 존재하며, 다음과 같이 정의된다:
 $$V^{*}(\mathrm{s}) = \max_{\pi}V^{\pi}(\mathrm{s}) \quad \forall \mathrm{s} \in \mathcal{S}$$

여기서 **만약 우리가 $V^{*}(\mathrm{s})$를 알고 있다면**, <mark><strong>최적 정책</strong></mark>은 상태 $\mathrm{s}_{t}$에서 가능한 모든 행동들 중에서 다음 상태 $\mathrm{s}_{t+1} \sim \mathcal{T}(\mathrm{s}_{t+1}\mid \mathrm{s}_{t}, a)$를 고려하여 **기댓값** $\mathbb{E}[V^{*}(\mathrm{s}_{t+1})]$을 최대화하는 행동 $a$를 선택함으로써 도출할 수 있다. 이 해설은 아래 수식으로 표현된다:
$$\pi^{*}(\mathrm{s}_{t})=\arg\max_{a}\mathbb{E}_{\mathrm{s}_{t+1} \sim \mathcal{T}(\mathrm{s}_{t+1}\mid \mathrm{s}_{t}, a)} [V^{*}(\mathrm{s}_{t+1})].$$

강화학습 또는 MDP 설정에서, 일반적으로 **전이 동역학 (상태 전이확률) $\mathcal{T}$은  알기 어렵거나 정확히 알 수 없다.** 따라서 우리는 또 다른 가치 함수인 **상태-행동 가치 함수(State-Action Value Function) 또는 Q 함수** $Q^{\pi}(s, a)$를 정의한다.
이는 $V^{\pi}(s)$와 유사하지만, 차이점은 처음 행동 $a$를 명시하고, 그 다음 상태부터는 정책 $\pi$를 따르는 것이다:
 $$Q^{\pi}(s,a)=\mathbb{E}[R\mid s,a,\pi]$$
이러한 $Q^{\pi}(s,a)$가 주어졌을 때 가장 좋은 정책을 구성하려면,  매 상태마다 **Greedy 방식으로 최대 순시보상 값을 주는 행동 $a$를 선택하여 가장 좋은 정책을 구성**할 수 있다:
$$\pi(s)=\arg\max_{a}Q^{\pi}(s,a)$$
또한, 해당 정책 하에서의 가능한 모든 행동에 대한 $Q^{\pi}(s,a)$ 값 중 최대값이 곧 상태 $s$의 가치 $V^{\pi}(s)$가 된다.
($Q \rightarrow V$의 환원 관계):
$$V^{\pi}(s)= \max_{a} Q^{\pi}(s,a)$$
 
 **Dynamic Programming:**  실제로 $Q^{\pi}$를 학습하기 위해 우리는 **마르코프 특성**을 활용하여 다음과 같은 **벨만 방정식(Bellman Equation)**형태로 재귀적으로 정의한다:
 $$Q^{\pi}(s_{t}, a_{t})=\mathbb{E}_{s_{t+1}}[r_{t+1}+\gamma Q^{\pi}(s_{t+1}, \pi(s_{t+1}))]$$
이는 $Q^{\pi}$를 **부트스트래핑(bootstrapping)** -- 즉, 현재 추정값을 사용해 더 나은 추정값을 만들 수 있음을 의미한다. 이러한 아이디어가 **Q-러닝** 및 **SARSA(State-Action-Reward-State-Action)** 알고리즘의 토대이다:
$$ Q^{\pi}(s_{t}, a_{t}) \leftarrow Q^{\pi}(s_{t}, a_{t}) +\alpha \delta $$
여기서 $\alpha$는 학습률, $\delta = Y-Q^{\pi}(s_{t}, a_{t})$는 **시간차-차이(Temporal Difference) 오차**이며, $Y$는 회귀 문제의 **타깃 값**과 같다.
- **SARSA (on-policy learning)**
	- 행동 정책(현재 $Q ^{\pi}$에서 파생된 정책)으로부터 생성된 전이를 사용해 $Q^{\pi}$를 개선한다. 이 경우:
	$$Y= r_{t+1}+\gamma Q^{\pi}(s_{t+1}, a_{t+1})$$
- **Q-Learning (off-policy learning)**
	- 전이가 파생 정책에 의해 생성되지 않아도 $Q^{\pi}$를 업데이트할 수 있다. 타깃을 아래와 같이 설정하여 곧바로 $Q^{*}$를 근사한다.
	$$ Y=r_{t+1}+\gamma \max_{a} Q^{\pi}(s_{t+1}, a)$$

임의의 $Q^{\pi}$로부터 **최적의 행동 가치 함수 $Q^{*}$를 찾으려면, 일반화된 정책 반복(Generalized Policy Iteration; GPI)를 사용**한다. 정책 반복의 경우, **정책 평가(Evaluation)와 정책 개선(Improvement)** 두 단계로 이루어 진다. **전통적인 정책 반복은 평가와 개선 단계를 수렴할 때까지 분리해 수행**하지만, **GPI는 두 단계를 교차(Interleave) 시켜 더 빠른 개선과 더 높은 평가를 통해 최적 정책 탐색**을 진행한다. 
- 정책 평가 (Policy Evaluation)
	- 에이전트가 경험한 **궤적(Trajectory)들의 TD 오차를 최소화**하여 가치 함수를 개선
- 정책 개선 (Policy Improvement)
	- 업데이트된 가치 함수에 **탐욕적(Greedy)** 방향으로 행동을 선택해 정책을 향상

### 3. B. Sampling
**부트스트래핑(동적 계획법) 대신, 몬테카를로(Monte Carlo) 방법**은 하나의 상태에서 정책을 여러 번 롤아웃(에피소드 실행)해 얻은 반환을 평균하여 **기대 반환**을 추정한다. 이 덕분에 순수 몬테카를로 기법은 **비-마르코프(Non-Markovian) 환경**에서도 적용 가능하다. 반면, 반환값을 계산하려면 롤아웃이 반드시 끝나야 하므로 에피소드MDP 에서만 사용할 수 있다는 제약이 있다. 이러한 부트스트래핑과 몬테카를로 기법을 절충한 것이 $TD(\lambda)$ 알고리즘이다.
TD학습과 몬테카를로 평가를 혼합하며, 할인 인자 $\gamma$와 유사하게 $\lambda$가 몬테카를로 방식($\lambda \approx 1$)과 부트스트래핑 방식($\lambda \approx 0$) 사이를 보간한다. Figure 3에서 보여주듯, 샘플링 양에 따라 **연속적인 RL 방법의 스펙트럼**이 형성된다.
또 다른 주요 가치 함수 기반 접근법은 **어드밴티지 함수 $A^{\pi}(s,a)$ 학습**이다. $Q^{\pi}$는 절대적 상태-행동 가치를 산출하는 데 비해, $A^{\pi}$는 **상대적 가치**를 표현한다.
$$A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)$$
이는 신호에서 **평균치(베이스라인)를 제거해 변동만 남기는 것**과 유사한 접근이다. 직관적으로, 어떤 행동이 다른 행동보다 **얼마나 더 나은지(상대적 편차)**를 학습하는 편이 행동 자체의 절대 반환을 직접 추정하는 것보다 쉬운 경우가 많다. 이 개념은 그라디언트 기반 정책 탐색에서 **분산 감소**를 위한 베이스라인 기법과도 밀접하다. 이러한 어드밴티지 업데이트 아이디어는 최근 DRL 알고리즘들에서 폭 넓게 활용되어지고 있다.

<div style="text-align: center">
    <figure>
        <img src="/assets/images/posts/a-brief-survey-of-deep-reinforcement-learning/fig3.png" alt="Reinforcement Learning Loop">
        <figcaption  style="text-align: center">Figure 3</figcaption>
    </figure>
</div>

<em>:> Ref.</em>

> Figure 3에 나타나는 "BACKUPS"은 값 추정을 갱신할때 '미래로부터 얼마만큼의 정보를 되돌려(Back-up) 반영하느냐' 라는 맥락으로 쓰인다.
 
| 구분 | 백업(Back-up) 유형 | 계산 방식 | 관련 알고리즘 |
|------|-----------|---------------|-------------------|
| **(a) Dynamic Programming** | **Full + Shallow**<br>(모든 후속 상태 기댓값 • 1‑step) | 다음 상태 분포 전체를 합산한 **정확한 기대값**으로 1‑스텝 부트스트랩 | Value Iteration, Policy Iteration |
| **(b) Exhaustive Search** | **Full + Deep**<br>(모든 후속 상태 기댓값 • 여러 스텝/트리 끝) | 전체 결정 트리를 확장·평가해 **모든 경로의 보상 합산** | Minimax / Expectimax Tree |
| **(c) Temporal‑Difference (TD)** | **Sample + Shallow**<br>(단일 샘플 • 1‑step) | 한 번 샘플링한 \(s_{t+1}\) 의 **TD 오차**로 즉시 갱신 | TD(0), Q‑Learning, SARSA |
| **(d) Monte Carlo** | **Sample + Deep**<br>(단일 샘플 궤적 • 에피소드 종료까지) | 한 에피소드 전체 리턴을 **평균하여** 값 추정 | Every‑Visit / First‑Visit Monte Carlo |

### 3. C. Policy Search

정책 탐색(Policy Search) 방법은 **가치 함수 모델을 유지하지 않고** 곧바로 최적 정책 $\pi^{*}$를 찾는다. 일반적으로 **매개변수화된 정책** $\pi_{\theta}$를 정의한 뒤, 파라미터 $\theta$를 조정해 **기대 반환** $\mathbb{E}[R\mid \theta]$을 최대화한다.
최적화는 **그래디언트 기반** 또는 **그래디언트-프리** 기법으로 수행된다.
- Gradient-free optimisation
	- 진화 전략, 유전 알고리즘 등 휴리스틱 탐색으로 **저차원 파라미터 공간**을 효과적으로 커버
	- 대규모 네트워크에도 일부 성공 사례가 있으나, 일반적으로는 샘플 효율이 떨어짐
- Gradient-based optimisation
	- Policy-gradient 기법들 (REINFORCE, TRPO, PPO 등)
	- **고차원 파라미터**를 가진 신경망 정책에 더 샘플-효율적이어서 대부분의 DRL 알고리즘에서 표준 선택

정책을 직접 구성할 때는 **확률 분포의 파라미터**를 출력하는 경우가 많다.
- 연속 행동 → 정규분포의 **평균 + 표준편차**
- 이산 행동 → 다항 분포의 **각 행동 확률**

이렇게 하면 **확률적 정책**이 되어 행동을 직접 샘플링할 수 있다. 반면, Gradient-free 방법은 미분 불가능한 정책까지 최적화가 가능하다는 것이 큰 장점이다.

**Policy Gradients:**  그래디언트는 매개변수화(parameterize)된 정책을 개선하는 데 강력한 학습 신호를 제공한다. 그러나 기대 반환(Expectation of Return, $\mathbb{E}[R]$)을 계산하려면, 현재 정책 파라미터가 만들어 내는 '그럴듯한' 궤적(trajectory)들에 대해 평균을 구해야 한다. 이 평균화는 **결정론적 근사**(예: linearisation) 또는 **샘플링을 통한 확률적 근사**가 필요하다. 결정론적 근사는 환경의 전이 확률이 존재할 때 **(Model-based, known MDP)** 적용 가능하다. 보다 일반적인 **model-free (unknown MDP)** RL에서는 $\mathbb{E}[R]$(기대 반환)을 몬테카를로 방식으로 추정한다.
하지만, 몬테카를로 근사는 확률적 함수의 샘플에 의존적이기 때문에, 그래디언트가 그 샘플을 "통과"할 수 없는 문제가 발생한다. -> 무슨 문제?  


## 4. VALUE FUNCTIONS

  

## 5. POLICY SEARCH