---
layout: post
title: A Brief Survey of Deep Reinforcement Learning
date: 2025-04-08 14:58 +0800
tags: [Reinforcement Learning, Deep Learning, Survey]
toc:  true
comments: true
---

## OUTLINE

<strong>심층 강화 학습(Deep Reinforcement Learning; DRL)</strong>은 인공 지능 분야에 혁명을 일으킬 태세를 갖추고 있으며, 특히 시각 정보를 다루는 영역에서의 더 높은 수준의 이해를 통해 자율 시스템 구축을 향한 발자국을 나타내는 중이다.<br>
현재 <strong>딥러닝(Deep Learning; DL)</strong>은 <strong>강화학습(Reinforcement Learning;RL)</strong>이 픽셀영역에서 직접 비디오 게임을 배우는 것과 같이, 이전에는 다루기 힘들었던 문제로 확장할 수 있도록 지원하고 있다. 이러한 <strong>DRL</strong> 알고리즘은 로봇 공학에도 적용되어 실제 세계에서 카메라 입력을 통해 로봇 제어 정책을 직접 학습할 수 있게 한다. <br>
해당 Paper에서는 <strong>RL</strong> 일반 분야에 대한 소개로 시작하여, <mark>가치 기반 방법</mark>과 <mark>정책 기반 방법</mark>의 주요 흐름으로 내용을 서술한다. 또한, 본 Paper에서는 <strong>Deep Q-Network (DQN)</strong>, <strong>Trust Region Policy Optimization (TRPO)</strong> 및 <strong>Asynchronous Advantage Actor Critic (A3C)</strong>을 포함한 <strong>DRL</strong>의 핵심 알고리즘을 다룬다. 이와 병행하여 <strong>Deep Neural Network (DNN)</strong>의 고유한 장점을 강조하고 <strong>RL</strong>을 통한 시각적 이해에 중점을 둔다. <br>
결론적으로 <strong>RL</strong>, <strong>DRL</strong>, <strong>(DRL + DL)</strong> 분야 내의 여러 연구 영역을 포함하여 설명한다.<br>

## INTRODUCTION

AI 분야의 주요 목표 중 하나는 환경과 상호 작용하여 최적의 행동을 학습하고 시행 착오를 통해 시간이 지남에 따라 개선되는 완전 자율 에이전트를 생산하는 것이다. 대응력이 뛰어나고 효과적으로 학습할 수 있는 AI 시스템을 제작하는 것은 로봇(주변 세계를 감지하고 반응할 수 있음)에서 <strong>자연어</strong> 및 <strong>멀티미디어</strong>와 상호 작용할 수 있는 <strong>순수 소프트웨어 기반 에이전트</strong>에 이르기까지 오랜 과제였다. <br>
<mark>경험 기반 자율 학습을 위한 원칙적인 수학적 프레임워크</mark>는 <strong>RL</strong> [78]이다. <strong>RL</strong>은 과거에 어느 정도 성공을 거두었지만 [31], [53], [74], [81], 이전 접근 방식은 <mark>확장성이 부족</mark>했고 본질적으로 <mark>상당히 낮은 차원의 문제로 제한</mark>되었다. 이러한 제한 사항은 <strong>RL 알고리즘</strong>이 <strong>다른 알고리즘</strong>과 동일한 복잡성 문제를 공유하기 때문에 존재합니다. 즉, <strong>메모리 복잡성</strong>, <strong>계산 복잡성</strong> 및 기계 학습 알고리즘의 경우 <strong>샘플 복잡성</strong> [76]이다. <br>
최근 몇 년 동안 우리가 목격한 것은 <strong>DNN</strong>의 <mark>강력한 함수 근사 및 표현 학습 속성에 의존적인 딥러닝의 부상</mark>으로, 이러한 복잡성 문제를 극복할 수 있는 새로운 도구를 제공했다. <strong>DL</strong>의 출현은 <strong>머신러닝 (Machine Learning; ML)</strong>의 여러 분야에 상당한 영향을 미쳐 객체 감지, 음성 인식 및 언어 번역과 같은 작업에서 최첨단 기술을 획기적으로 향상시켰다 [39]. <br>
<strong>DL</strong>의 가장 중요한 속성은 <strong>DNN</strong>이 이미지, 텍스트 및 오디오와 같은 고차원 데이터의 압축된 <strong>저차원 표현(특징)</strong>을 자동으로 찾을 수 있다는 것이다. 신경망 아키텍처, 특히 계층적 표현에 <strong>귀납적 편향</strong>을 적용함으로써 <strong>ML 실무자</strong>들은 <mark>차원의 저주</mark>를 해결하는 데 효과적인 진전을 이루었다 [7]. <br>
이러한 <strong>DL</strong>의 기능은 <strong>RL</strong>의 발전을 가속화했으며, <strong>RL</strong> 내에서 <strong>DL</strong> 알고리즘을 사용하면 비로소 <strong>DRL</strong> 필드가 정의된다. 본 Paper의 목표는 <strong>DRL</strong>의 획기적인 발전과 최근(2017)의 발전을 모두 다루고, 신경망이 자율 에이전트 개발에 더 가까워지도록 사용될 수 있는 혁신적인 방법을 전달하는 것이다. <em>DRL의 최근 노력에 대한 보다 포괄적인 Survey Paper는 Li [43]의 개요를 참조.</em><br>
