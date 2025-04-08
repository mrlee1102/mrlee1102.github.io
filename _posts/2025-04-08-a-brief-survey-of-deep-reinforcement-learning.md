---
layout: post
title: A Brief Survey of Deep Reinforcement Learning
date: 2025-04-08 14:58 +0800
tags: [Reinforcement Learning, Deep Learning, Survey]
toc:  true
comments: true
---

## OUTLINE

<mark>심층 강화 학습(Deep Reinforcement Learning; DRL)</mark>은 인공 지능 분야에 혁명을 일으킬 태세를 갖추고 있으며, 특히 시각 정보를 다루는 영역에서의 더 높은 수준의 이해를 통해 자율 시스템 구축을 향한 발자국을 나타내는 중이다. 현재 <mark>딥러닝(Deep Learning; DL)</mark>은 <mark>강화학습(Reinforcement Learning;RL)</mark>이 픽셀영역에서 직접 비디오 게임을 배우는 것과 같이, 이전에는 다루기 힘들었던 문제로 확장할 수 있도록 지원하고 있다. 이러한 <mark>DRL</mark> 알고리즘은 로봇 공학에도 적용되어 실제 세계에서 카메라 입력을 통해 로봇 제어 정책을 직접 학습할 수 있게 한다. 해당 Paper에서는 <mark>RL</mark> 일반 분야에 대한 소개로 시작하여, <strong>가치 기반 방법</strong>과 <strong>정책 기반 방법</strong>의 주요 흐름으로 내용을 서술한다. 또한, 본 Paper에서는 <mark>Deep Q-Network (DQN)</mark>, <mark>Trust Region Policy Optimization (TRPO)</mark> 및 <mark>Asynchronous Advantage Actor Critic (A3C)</mark>을 포함한 <mark>DRL</mark>의 핵심 알고리즘을 다룬다. 이와 병행하여 <mark>Deep Neural Network (DNN)</mark>의 고유한 장점을 강조하고 <mark>RL</mark>을 통한 시각적 이해에 중점을 둔다. 결론적으로 <mark>RL</mark>, <mark>DRL</mark>, <mark>(DRL + DL)</mark> 분야 내의 여러 연구 영역을 포함하여 설명한다.

## INTRODUCTION

AI 분야의 주요 목표 중 하나는 환경과 상호 작용하여 최적의 행동을 학습하고 시행 착오를 통해 시간이 지남에 따라 개선되는 완전 자율 에이전트를 생산하는 것이다. 대응력이 뛰어나고 효과적으로 학습할 수 있는 AI 시스템을 제작하는 것은 로봇(주변 세계를 감지하고 반응할 수 있음)에서 <mark>자연어</mark> 및 <mark>멀티미디어</mark>와 상호 작용할 수 있는 <mark>순수 소프트웨어 기반 에이전트</mark>에 이르기까지 오랜 과제였다. <strong>경험 기반 자율 학습을 위한 원칙적인 수학적 프레임워크</strong>는 <mark>RL</mark> [78]이다. <mark>RL</mark>은 과거에 어느 정도 성공을 거두었지만 [31], [53], [74], [81], 이전 접근 방식은 <strong>확장성이 부족</strong>했고 본질적으로 <strong>상당히 낮은 차원의 문제로 제한</strong>되었다. 이러한 제한 사항은 <mark>RL 알고리즘</mark>이 <mark>다른 알고리즘</mark>과 동일한 복잡성 문제를 공유하기 때문에 존재합니다. 즉, <mark>메모리 복잡성</mark>, <mark>계산 복잡성</mark> 및 기계 학습 알고리즘의 경우 <mark>샘플 복잡성</mark> [76]이다. 최근 몇 년 동안 우리가 목격한 것은 <mark>DNN</mark>의 <strong>강력한 함수 근사 및 표현 학습 속성에 의존적인 딥 러닝의 부상</strong>으로, 이러한 문제를 극복할 수 있는 새로운 도구를 제공했다. <mark>DL</mark>의 출현은 <mark>머신러닝 (Machine Learning; ML)</mark>의 여러 분야에 상당한 영향을 미쳐 객체 감지, 음성 인식 및 언어 번역과 같은 작업에서 최첨단 기술을 획기적으로 향상시켰다 [39]. <mark>DL</mark>의 가장 중요한 속성은 <mark>DNN</mark>이 이미지, 텍스트 및 오디오와 같은 고차원 데이터의 압축된 <mark>저차원 표현(특징)</mark>을 자동으로 찾을 수 있다는 것이다. 신경망 아키텍처, 특히 계층적 표현에 <mark>귀납적 편향</mark>을 적용함으로써 <mark>ML 실무자</mark>들은 <strong>차원의 저주</strong>를 해결하는 데 효과적인 진전을 이루었다 [7]. <mark>DL</mark>은 기능은 <mark>RL</mark>의 발전을 가속화했으며, <mark>RL</mark> 내에서 <mark>DL</mark> 알고리즘을 사용하면 비로소 <mark>DRL</mark> 필드가 정의된다. 본 Paper의 목표는 <mark>DRL</mark>의 획기적인 발전과 최근(2017)의 발전을 모두 다루고, 신경망이 자율 에이전트 개발에 더 가까워지도록 사용될 수 있는 혁신적인 방법을 전달하는 것입니다. <em>DRL의 최근 노력에 대한 보다 포괄적인 Survey Paper는 Li [43]의 개요를 참조</em>하시오.
