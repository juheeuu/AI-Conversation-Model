# 삼성 DS-KAIST AI Expert 프로그램 
## Conversation Model

실습 일시: 2019년 11월 29일 (수), 13:00 - 18:00

담당 조교: 박진영 (jy.bak@kaist.ac.kr), 박성준 (sungjoon.park@kaist.ac.kr)


### Tasks

- VHRED 코드 완성
	- ``/src/models/vhred.py`` 파일 안 `prior` 함수 구현
	- 각 함수가 해야할 역할은 코드 및 pdf 파일 참고
- HRED와 VHRED 실행
	- 모델 학습
	- 대답 생성
	- 생성된 대답 평가
- 대화 대답 평가방법 논의
	- 소개된 3개의 평가방법에 대한 한계점
	- 대화 대답 평가방법 제안

### Hints
```
def posterior(self, context_outputs, encoder_hidden):
        """
        Compute variational posterior
        :param context_outputs: h_t^{cxt} [num_true_utterances, context_rnn_output_size]
        :param encoder_hidden: x_t [num_true_utterances, encoder_rnn_output_size]
        :return: [mu, sigma]
        """
        h_posterior = self.posterior_h(torch.cat([context_outputs, encoder_hidden], 1))
        mu_posterior = self.posterior_mu(h_posterior)
        var_posterior = self.softplus(self.posterior_var(h_posterior))
        return mu_posterior, var_posterior
```
