%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
function Figure1F_20230531_get_optimal_threshold_reward_task

x_evi_plot = [0:0.01:1];

stim_para = [0.5 0.25 0.25 0.25 0.25 0.5];
stim_prob = [0.5 0.5];
stim_prob2 = [0.5 0.5];
reward = [3,1];
reward2 = [1,3];

use_uncertainty = [
7.747
6.908
6.7961
3.9777
4.1662
5.2626
5.0882
7.4139
4.6815
7.0278
4.5463
6.8968
5.4891
7.7503
6.4555
7.083
6.2636
4.8833
6.528
7.366
5.6101
7.053
6.741
5.4557
7.967
7.053
7.7955
6.0181
6.8246
5.8546
5.3621
5.2489
5.8181
4.3704
5.9956
5.9073
6.5569
4.9383
5.0948
6.0029
6.8223
6.4275
4.3273
5.6753
4.7272
4.4404
3.7682
3.3685
6.7997
4.4904
6.0005
6.3415
6.1294
4.3309
5.4152
5.7814
7.2965
5.9551
5.7411
6.6723
6.1148
6.6005
5.7732
5.1147
4.7947
5.4501
5.2716
5.3587
7.3263
5.4312
5.3878
4.9197
4.9173
6.894
5.5816
6.0669
4.1125
6.1385
5.4445
7.1421
4.9274
4.1067
5.7774
];

opt = optimset('Display','off');
for session = 1:length(use_uncertainty),

    clear X_all FCAL_all
    for i = 1 : 100,
        [X_all(i,:),FCAL_all(i),EXITFLAG,OUTPUT] = fminsearch(@find_max_psycho,rand,opt,...
            use_uncertainty(session), stim_para, stim_prob, reward);
    end
    temp_FCAL = find(FCAL_all == min(FCAL_all),1);
    para_max = X_all(temp_FCAL,:);
    clear X_all FCAL_all
    for i = 1 : 100,
        [X_all(i,:),FCAL_all(i),EXITFLAG,OUTPUT] = fminsearch(@find_max_psycho,rand,opt,...
            use_uncertainty(session), stim_para, stim_prob2, reward2);
    end
    temp_FCAL = find(FCAL_all == min(FCAL_all),1);
    para_max2 = X_all(temp_FCAL,:);

    para(session,:) = [para_max, para_max2];

    psychometric(session,:)  = 1./(1+exp(-use_uncertainty(session) .* x_evi_plot + para_max));
    psychometric2(session,:) = 1./(1+exp(-use_uncertainty(session) .* x_evi_plot + para_max2));
    psychometric05(session,:) = 1./(1+exp(-use_uncertainty(session) .* x_evi_plot + use_uncertainty(session)/2));
end

mean_psycho1 = mean(psychometric,2);
mean_psycho2 = mean(psychometric2,2);

sabun_psycho = mean_psycho2 - mean_psycho1

figure
plot(x_evi_plot, mean(psychometric,1), 'b')
hold on
plot(x_evi_plot, mean(psychometric2,1), 'r')
hold on
plot(x_evi_plot, mean(psychometric05,1), 'k')

set(gca,'xlim',[-0.1 1.1],'ylim',[0 1])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function r = find_max_psycho(para, use_uncertainty, stim_para, stim_prob, reward)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r(1) = reward(1) .* stim_prob(1) .* stim_para(1) .* (1-1/(1+exp(para)));
r(2) = reward(1) .* stim_prob(1) .* stim_para(2) .* (1-1/(1+exp(-0.25*use_uncertainty+para)));
r(3) = reward(1) .* stim_prob(1) .* stim_para(3) .* (1-1/(1+exp(-0.45*use_uncertainty+para)));
r(4) = reward(2) .* stim_prob(2) .* stim_para(4) .* (1/(1+exp(-0.55*use_uncertainty+para)));
r(5) = reward(2) .* stim_prob(2) .* stim_para(5) .* (1/(1+exp(-0.75*use_uncertainty+para)));
r(6) = reward(2) .* stim_prob(2) .* stim_para(6) .* (1/(1+exp(-use_uncertainty+para)));

r = -sum(r);

return


