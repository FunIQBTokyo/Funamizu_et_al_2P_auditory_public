%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
function Figure1F_20230531_get_optimal_threshold_stimulus_task

x_evi_plot = [0:0.01:1];

stim_para = [0.5 0.25 0.25 0.25 0.25 0.5];
stim_prob = [0.7 0.3];
stim_prob2 = [0.3 0.7];
reward = [2,2];
reward2 = [2,2];

use_uncertainty = [
4.7874
6.5522
8.41
8.048
3.9329
3.6197
4.3347
5.0151
7.1098
5.6319
3.2158
5.7309
4.9127
7.6506
8.5673
5.1791
5.9124
5.2212
6.455
5.3615
6.6568
7.4068
6.353
7.7687
7.8504
6.9061
8.4104
7.2161
6.2644
7.527
7.1926
4.9469
6.2975
7.1953
7.84
6.6624
6.3804
4.9609
5.2918
5.1349
5.4074
5.3776
5.6518
4.0034
5.2269
6.9994
6.1171
3.8773
5.0209
5.3762
4.5591
5.3346
7.5114
3.9884
4.8868
5.0693
5.4695
6.9581
6.653
6.099
7.3326
7.6136
6.3638
3.9027
5.2941
5.4347
6.6418
6.8271
6.8676
5.4955
6.3261
5.4928
6.1194
6.8144
6.6314
6.3973
5.4654
5.0864
6.4072
7.5865
5.5173
8.1334
6.3146
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

