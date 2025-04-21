---
layout: post
title: A Brief Survey of Deep Reinforcement Learning
date: 2025-04-20 14:30 +0800
tags: [Reinforcement Learning, Deep Learning, Survey]
toc: true
comments: true
math: true
---

## Abstract

<strong>심층 강화 학습(Deep Reinforcement Learning; DRL)</strong>은 인공 지능 분야에 혁명을 일으킬 태세를 갖추고 있으며, 특히 시각 정보를 다루는 영역에서의 더 높은 수준의 이해를 통해 자율 시스템 구축을 향한 발자국을 나타내는 중이다.<br  />

현재 <strong>딥러닝(Deep Learning; DL)</strong>은 <strong>강화학습(Reinforcement Learning;RL)</strong>이 픽셀영역에서 직접 비디오 게임을 배우는 것과 같이, 이전에는 다루기 힘들었던 문제로 확장할 수 있도록 지원하고 있다. 이러한 DRL 알고리즘은 로봇 공학에도 적용되어 실제 세계에서 카메라 입력을 통해 로봇 제어 정책을 직접 학습할 수 있게 한다. <br  />

해당 Paper에서는 RL 일반 분야에 대한 소개로 시작하여, <strong>가치 기반 방법</strong>과 <strong>정책 기반 방법</strong>의 주요 흐름으로 내용을 서술한다. 또한, 본 Paper에서는 <strong>Deep Q-Network (DQN)</strong>, <strong>Trust Region Policy Optimization (TRPO)</strong> 및 <strong>Asynchronous Advantage Actor Critic (A3C)</strong>을 포함한 DRL의 핵심 알고리즘을 다룬다. 이와 병행하여 <strong>Deep Neural Network (DNN)</strong>의 고유한 장점을 강조하고 RL을 통한 시각적 이해에 중점을 둔다. <br  />

결론적으로 RL, DRL, DRL + DL 분야 내의 여러 연구 영역을 포함하여 설명한다.<br  />

## 1. INTRODUCTION

AI 분야의 주요 목표 중 하나는 환경과 상호 작용하여 최적의 행동을 학습하고 시행 착오를 통해 시간이 지남에 따라 개선되는 완전 자율 에이전트를 생산하는 것이다. 대응력이 뛰어나고 효과적으로 학습할 수 있는 AI 시스템을 제작하는 것은 로봇(주변 세계를 감지하고 반응할 수 있음)에서 <strong>자연어</strong> 및 <strong>멀티미디어</strong>와 상호 작용할 수 있는 <strong>순수 소프트웨어 기반 에이전트</strong>에 이르기까지 오랜 과제였다. <br  />

<strong>경험 기반 자율 학습을 위한 원칙적인 수학적 프레임워크</strong>는 RL이다. RL은 과거에 어느 정도 성공을 거두었지만, 전통적인 접근 방식은 <strong>확장성이 부족</strong>했고 본질적으로 <strong>상당히 낮은 차원의 문제로 제한</strong>되었다. 이러한 제한 사항은 RL 알고리즘이 다른 알고리즘과 동일한 복잡성 문제를 공유하기 때문에 존재합니다. 즉, <strong>메모리 복잡성</strong>, <strong>계산 복잡성</strong> 및 기계 학습 알고리즘의 경우 <strong>샘플 복잡성</strong>이다. <br  />

최근 몇 년 동안 우리가 목격한 것은 DNN의 <strong>강력한 함수 근사 및 표현 학습 속성에 의존적인 딥러닝의 부상</strong>으로, 이러한 복잡성 문제를 극복할 수 있는 새로운 도구를 제공했다. DL의 출현은 <strong>머신러닝 (Machine Learning; ML)</strong>의 여러 분야에 상당한 영향을 미쳐 객체 감지, 음성 인식 및 언어 번역과 같은 작업에서 최첨단 기술을 획기적으로 향상시켰다. <br  />

  

DL의 가장 중요한 속성은 DNN이 이미지, 텍스트 및 오디오와 같은 고차원 데이터의 압축된 <strong>저차원 표현(특징)</strong>을 자동으로 찾을 수 있다는 것이다. 신경망 아키텍처, 특히 계층적 표현에 <strong>귀납적 편향</strong>을 적용함으로써 <strong>ML 실무자</strong>들은 <strong>차원의 저주</strong>를 해결하는 데 효과적인 진전을 이루었다. <br  />

이러한 DL의 기능은 RL의 발전을 가속화했으며, RL 내에서 DL 알고리즘을 사용하면 비로소 DRL 필드가 정의된다. 본 Paper의 목표는 DRL의 획기적인 발전과 최근(2017)의 발전을 모두 다루고, 신경망이 자율 에이전트 개발에 더 가까워지도록 사용될 수 있는 혁신적인 방법을 전달하는 것이다. <em>DRL의 최근 노력에 대한 보다 포괄적인 Survey Paper는 Li [43]의 개요를 참조.</em><br  />

DL은 RL이 이전에 다루기 어려웠던 <strong>의사 결정 문제, 즉 고차원 상태 및 액션 공간을 가진 설정으로 확장</strong>할 수 있도록 기여한다. DRL 분야의 최근 연구 중 두 가지 뛰어난 성공 사례가 있다. DRL의 혁명을 시작한 <strong>첫 번째</strong>는 이미지 픽셀에서 직접 Atari 2600 비디오 게임을 초인적인 수준으로 플레이하는 방법을 배울 수 있는 알고리즘의 개발이었다. RL에서 함수 근사 기술의 불안정성에 대한 솔루션을 제공하는 이 연구는 <strong>RL 에이전트가 보상 신호만을 기반으로 원시적인 고차원 관찰에 대해 훈련될 수 있음</strong>을 설득력 있게 입증한 최초의 연구였다. <br  />

<strong>두 번째 뛰어난 성공</strong>은 IBM의 Deep Blue가 20년 전에 체스에서 거둔 역사적인 업적과 유사하게, 인간 세계 챔피언을 바둑에서 물리친 <strong>하이브리드 DRL 시스템인 AlphaGo</strong>의 개발이다 [9]. 체스 게임 시스템을 지배했던 수제 규칙과 달리 AlphaGo는 <strong>지도 학습 및 RL을 사용하여 훈련된 신경망과 기존의 휴리스틱 검색 알고리즘을 결합하여 구성</strong>되었다. <br  />

이러한 DRL 알고리즘은 로봇 공학과 같이 광범위한 문제에 이미 적용되었으며, 로봇의 제어 정책은 이제 실제 세계의 카메라 입력을 통해 직접 학습할 수 있다. 기존에는 직접 설계하거나 로봇 상태의 저차원 특징으로부터 학습되었던 컨트롤러의 후속이 된다. <br  />

## 2. Reward-driven Behavior

DNN이 RL에 기여하는 바를 살펴보기 전에 RL 분야 전반은 <strong>  <strong> 상호작용을 통한 학습</strong>  </strong>에 뿌리를 두고 있다. RL 에이전트는 환경과 상호작용하여, 자신의 행동 결과를 관찰<strong>(Observation)</strong>하고 환경적 요소로 정의된 보상 규칙에 대한 보상 값에 따라 자신의 행동을 변화하는 법을 학습한다. 이러한 상호작용(시행착오)를 거듭한 학습 패러다임은 행동주의 심리학에 근간되며, RL의 기반이다. RL에 영향을 준 또다른 주요 분야는 최적 제어(Optimal Control)이며, 특히 동적 계획법 (Dynamic Programming; DP)과 같은 수학적 형식을 RL의 토대로 제공해 왔다. <br  />

RL 설정에서, 자율 <strong>에이전트</strong>는 ML 알고리즘에 의해 제어되며 시점(Timestamp) $t$에서 <strong>환경</strong>으로부터 <strong>상태 $s_t$</strong>를 관측(Observation)한다. 에이전트는 상태 $s_t$에서 <strong>행동 $a_t$</strong>을 수행하여 환경과 상호작용한다. 에이전트가 행동을 취한 뒤, 현재 상태와 선택된 행동에 따라 환경과 에이전트는 <strong>새로운 상태 $s_{t+1}$</strong>로 전이(Transition)한다. <br  />

상태는 환경에 대한 충분 통계(Sufficient Statistic)로서, 최적의 행동을 결정하는 데 필요한 모든 정보를 포함하고 있다. 여기에는 에이전트의 일부 (ex: 구동기(액추에이터) 및 센서의 위치)도 포함될 수도 있다. 최적 제어 문헌에서는 상태와 행동을 흔히 $x_t$와 $u_t$로 표기한다. <br  />

최적의 행동 순서는 환경이 제공하는 보상에 의해 결정된다. 환경이 새로운 상태로 전이할 때마다, 에이전트는 피드백으로 스칼라 보상 $r_{t+1}$을 받는다. <strong>에이전트의 목표</strong>는 기대 반환(할인된 보상의 누적값)을 극대화하는 정책(제어 전략) $\pi$를 학습하는 것이다. 상태가 주어지면 정책은 수행할 행동을 반환하며, '최적 정책'이란 환경에서 기대 반환을 최대화하는 모든 정책을 의미한다.

이런 점에서 강화학습(RL)은 최적 제어와 동일하게 문제를 해결하는 접근을 한다. 그러나 RL의 개척점은 최적 제어와 달리 상태 전이 동역학 모델이 주어지지 않고도, 에이전트가 시행착오를 통해 행동의 결과를 학습해야 한다는 데 있다. 환경과의 매 상호작용은 정보(Information, Source Data)를 제공하며, 에이전트는 이를 활용해 자신의 지식을 갱신한다. 이러한 관측(인지, 지각)‑행동‑학습 루프가 그림 2에 제시되어 있다. <br  />

<div  style="text-align: center">
    <figure>
        <img  src="/assets/images/posts/2025-04-20-a-brief-survey-of-deep-reinforcement-learning/fig2.png"  alt="Reinforcement Learning Loop">
        <figcaption  style="text-align: center">Figure 2</figcaption>
    </figure>
</div>

### 2. A. Markov Decision Process

형식적으로 RL은 마르코프 결정 과정(Markov Decision Process; MDP)으로 기술할 수 있으며, MDP는 다음 요소로 구성된다.

Elements of MDP: 
- 상태 집합 $\mathcal{S}$ 및 초기 상태 분포 $p(s_0)$
- 행동 집합 $\mathcal{A}$
- 전이 동역학 (상태 전이확률) $\mathcal{T}(s_{t+1} \mid s_t, a_t)$
- 순시(즉시, 순간) 보상 함수 $\mathcal{R}(s_t, a_t, s_{t+1})$
- 할인 인자 $\gamma \in [0,1]$ — 값이 0에 가까울수록 순시 보상(현재 시점 피드백)에 더 큰 가중치를 부여
<br />

일반적으로 <strong>정책 $\pi$</strong>는 상태를 행동 확률 분포에 대응시키는 Mapping(사상)을 의미한다.

$$ \pi: \mathcal{S} \rightarrow p(\mathcal{A}=a \mid \mathcal{S}) $$

MDP가 <strong>"에피소드형(Episodic)"</strong>이라면 (즉, 길이가 $T$ step인 에피소드가 끝날 때마다 상태가 리셋됨) 하나의 에피소드에서 얻는 상태-행동-보상의 연속적인 과정을 <strong>궤적(Trajectory) 또는 롤아웃(rollout)</strong>이라 칭한다. <strong>각 trajectory에서 누적되는 보상의 총합(즉, 에피소드동안 얻은 순시보상들의 총합)은 반환(return)</strong>이라 하며, 다음 $R$로 정의된다.

$$ 
R = \sum_{t=0}^{T-1} \gamma^t r_{t+1} 
$$

RL의 목표는 모든 상태에서 <strong>기대 반환</strong>을 최대화하는 최적 정책 $\pi^*$를 찾는 것이다.

$$ 
\pi^* = \arg\max_{\pi} \mathbb{E}[R \mid \pi] 
$$

또한, $T = \infty$인 "비-에피소드형(Non-episodic)" MDP도 고려할 수 있다. 이 경우에는 $\gamma < 1$이어야 보상의 무한합이 발산하지 않게 유한하도록 형성된다. (즉, $\gamma$를 1보다 작은 값으로 설정하면, 반복이 진행될수록 미래 보상의 기여도가 기하급수적으로 줄어들어 누적 보상 합이 수렴한다. 이는 무한‑수명 작업(ex: 자율주행 순찰)에서도 가치 함수가 안정적으로 평가 및 학습되도록 하기 위해 사용되는 할인 기법이다.)
<br /><br />

> <strong>Markov Property</strong>: 다음 <strong>상태는 오직 현재 상태에만 영향을 받으며</strong>, 달리 표현하면 <strong>현재 상태가 주어지면 미래는 과거와 조건부로 독립</strong>이다.

강화학습의 핵심 개념 중 하나는 <strong>마르코프 특성(Markov Property)</strong>이다. $t$시점의 상태인 $s_t$에서 내리는 의사 결정은 과거 상태 $\{s_0, s_1, \dots, s_{t-1}\}$ 전체가 아니라 <strong>직전 상태 $s_{t-1}$</strong>만을 기반으로 해도 충분하다는 의미이다. 대다수의 RL 알고리즘은 이 가정을 채택하지만, 이는 <strong>'상태가 완전히 관측 가능하다'</strong>는 이상적인 전제를 요구하기 때문에 현실적인 면에서 괴리가 존재한다. 따라서 이러한 MDP를 일반화한 개념인 <strong>부분 관측 마르코프 결정 과정 (Partially Observable MDP, POMDP)</strong>를 고려한다.

- POMDP에서는 에이전트가 <strong>관측(Observation) $o_t \in \Omega$</strong> 만을 수신한다.
- 관측의 분포 $p(o_{t+1} \mid s_{t+1}, a_t)$는 다음 상태 $s_{t+1}$와 직전 행동 $a_t$에 의존한다.

제어공학 및 신호처리 문맥에서 보자면, 이는 상태-공간 모델 (State-space model) 상의 측정/관측 매핑(measurement mapping)에 해당되며, 이 매핑 역시 <strong>현재 상태와 이전에 적용된 제어 입력(행동)</strong>에 의해 결정된다고 이해할 수 있다.
POMDP 알고리즘은 보통 <strong>이전 신념 상태, 수행한 행동, 현재 관측치</strong>를 바탕으로 현 상태에 대한 <strong>신념(*belief*)</strong>을 업데이트 한다. DL에서는 일반적으로 순환 신경망(Recurrent Neural Net; RNN)을 이용하는 접근이 널리 쓰인다. RNN은 순방향 신경망(Feedforward Neural Net; FNN)과 달리 <strong>동적 시스템</strong>이기 때문이다. 이러한 POMDP 해결 방식은, 실제 상태를 직접 관측할 수 없어 추정만 가능한 동적 시스템$\cdot$상태 공간 모델 문제들과 밀접하게 관련되어 있다.

### 2. B. Challenges in RL
