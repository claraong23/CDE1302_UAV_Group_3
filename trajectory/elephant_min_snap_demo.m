%% elephant_min_snap_demo
% Minimum-snap "elephant" path through a list of points, flown from the start.
%
% Frame: metres from the start point (0, 0); x = forward, y = right. The drone
% keeps its takeoff heading (nose +x) for the whole flight (trajectory mode).
%
% The path passes exactly through every point in `pts_cm`, in order, starting
% and ending at rest at (0, 0). Optional guide points (max_leg_m > 0) pull it
% onto the straight lines between the points; with enough points (as now) they
% are not needed and only make the edges wavy.
% One seventh-order polynomial per leg; velocity,
% acceleration and jerk are continuous at every point, and the speed and
% direction there are left free, so the optimiser picks the smoothest way
% through (minimum integrated snap).
%
% Leg times start in proportion to leg length, are then balanced so every leg
% peaks at about the same speed, and finally all scaled together so the peak
% speed is v_limit.
%
% Outputs beside this file:
%   elephant_min_snap_traj.csv   t,x,y,z (z = 0: holds the takeoff height)
%   elephant_min_snap.png        top-down path through the points, and speed
%
% Fly it (no --traj-rotate; x is already forward):
%   python laptop/tag_mission.py --drone-id 3 --trajectory trajectory/elephant_min_snap_traj.csv --takeoff-wait 16 --confirm

clc; close all;
traj_dir = fileparts(mfilename('fullpath'));

%% User parameters
% Points to pass through, in order (cm; x forward, y right; left = negative).
pts_cm = [  0    0;
          140  -90;
          200  -90;
          320    0;
          380  100;
          250  120;
          140  100;
           80  -20;
          140 -140;
          250 -160;
          380 -140;
          320    0;
          200   65;
          140   65;
            0    0];
max_leg_m   = 0;      % 0 = pass through the points only (smoothest). > 0 adds guide
                      % points every this many m along each leg, which pulls the path
                      % onto the straight lines but makes the edges slightly wavy;
                      % useful only when the points are sparse (v1 bulged ~0.7 m)
v_limit     = 0.297;  % peak speed (m/s); tag_mission.py refuses > 0.30
end_factor  = 1.5;    % first/last legs get this much more time (start/stop at rest)
balance_its = 15;     % passes that even out the legs' peak speeds
dt          = 0.01;   % output sample time (s); the uploader resamples

P_req = pts_cm / 100;                      % the points asked for
W = P_req(1, :);                           % ... plus guide points along each leg
pt_idx = 1;                                % row of W holding each requested point
for k = 1:size(P_req, 1) - 1
    seg = P_req(k+1, :) - P_req(k, :);
    n = 1;
    if max_leg_m > 0, n = max(1, ceil(norm(seg) / max_leg_m)); end
    W = [W; P_req(k, :) + (1:n).' / n * seg]; %#ok<AGROW>
    W(end, :) = P_req(k+1, :);             % exact point, no rounding
    pt_idx(end+1) = size(W, 1);            %#ok<AGROW>
end
M = size(W, 1) - 1;

%% Leg times: by length, then balanced, then scaled to the speed limit
T = vecnorm(diff(W), 2, 2).' / v_limit;
T([1 M]) = T([1 M]) * end_factor;
for it = 1:balance_its
    coeff = solveMinSnap(W, T);
    pk = legPeaks(coeff, T);
    T = T .* sqrt(pk / mean(pk));          % slow legs get less time, fast ones more
end
coeff = solveMinSnap(W, T);
T = T * max(legPeaks(coeff, T)) / v_limit; % uniform time scaling: same shape
coeff = solveMinSnap(W, T);

for pass = 1:3                             % rescale on the fine samples (exact peak)
    t = (0:dt:sum(T)).';
    if t(end) < sum(T), t(end+1) = sum(T); end %#ok<AGROW>
    [P, V] = sampleMinSnap(coeff, T, t);
    speed = vecnorm(V, 2, 2);
    if max(speed) <= v_limit, break; end
    T = T * max(speed) / v_limit * 1.0005;
    coeff = solveMinSnap(W, T);
end
P(end, :) = P_req(end, :);

fprintf('%d points (+ %d guide points), %.2f m of path\n', size(P_req, 1), ...
    size(W, 1) - size(P_req, 1), sum(speed(1:end-1) .* diff(t)));
fprintf('Points span x %.2f..%.2f, y %.2f..%.2f\n', ...
    min(P_req(:,1)), max(P_req(:,1)), min(P_req(:,2)), max(P_req(:,2)));
fprintf('Path spans x %.2f..%.2f, y %.2f..%.2f\n', min(P(:,1)), max(P(:,1)), min(P(:,2)), max(P(:,2)));
fprintf('Duration %.1f s; peak %.3f m/s; mean %.3f m/s\n', t(end), max(speed), mean(speed));
assert(max(speed) <= v_limit + 1e-6, 'peak speed above v_limit');

%% Write CSV (t, x, y, z)
csv_path = fullfile(traj_dir, 'elephant_min_snap_traj.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, '# Minimum-snap elephant path from elephant_min_snap_demo.m (x forward, y right, from the start)\n');
fprintf(fid, '# through (cm): %s; fly WITHOUT --traj-rotate\n', strjoin(compose('(%g,%g)', pts_cm), ' '));
fprintf(fid, '# %.1f s, peak %.3f m/s, dt %.3f s\n', t(end), max(speed), dt);
fprintf(fid, 't,x,y,z\n');
fprintf(fid, '%.3f,%.4f,%.4f,0\n', [t P].');
fclose(fid);
fprintf('Wrote %s\n', csv_path);

%% Plot: top view (x forward = up, y right = right) and speed
fig = figure('Name', 'Elephant min-snap', 'Color', 'w', 'Position', [100 100 1100 640]);
try, theme(fig, 'light'); catch, end
subplot(1, 2, 1);
plot(P(:,2), P(:,1), 'Color', [0.08 0.45 0.78], 'LineWidth', 2.5); hold on;
plot(P_req(:,2), P_req(:,1), 'ko', 'MarkerFaceColor', [0.95 0.55 0.1], 'MarkerSize', 7);
for k = 1:size(P_req, 1)
    text(P_req(k,2) + 0.07, P_req(k,1) + 0.07, sprintf('%d', k - 1), 'FontWeight', 'bold');
end
step = max(1, round(numel(t) / 35));
quiver(P(1:step:end-1,2), P(1:step:end-1,1), V(1:step:end-1,2), V(1:step:end-1,1), 0.5, ...
       'Color', [0.08 0.45 0.78]);
axis equal; grid on; box on;
xlabel('y right (m)'); ylabel('x forward (m)');
title(sprintf('Min-snap elephant, %.0f s (numbers = point order, 0 = start/land)', t(end)));
subplot(1, 2, 2);
plot(t, speed, 'LineWidth', 1.5); grid on; hold on;
cT = cumsum(T);                                        % time at each requested point
xline(cT(pt_idx(2:end-1) - 1), ':', 'Color', [0.6 0.6 0.6]);
xlabel('time (s)'); ylabel('speed (m/s)');
title(sprintf('peak %.3f m/s (dotted = passing a point)', max(speed)));
png_path = fullfile(traj_dir, 'elephant_min_snap.png');
exportgraphics(fig, png_path, 'Resolution', 130);
fprintf('Wrote %s\n', png_path);

%% Local functions
function coeff = solveMinSnap(W, T)
% Per axis: 8 coefficients per leg (normalised time tau in [0, 1]) minimising
% sum_m T_m^-7 * int (d4p/dtau4)^2 subject to: positions at both ends of every
% leg; zero velocity, acceleration and jerk at the start and end; velocity,
% acceleration and jerk continuous at every interior point.
M = numel(T);  N = 8 * M;
Q = zeros(N);
for m = 1:M
    idx = 8*(m-1) + (1:8);
    Q(idx, idx) = snapCost() / T(m)^7;
end
coeff = zeros(8, M, 2);
for ax = 1:2
    A = zeros(0, N);  b = zeros(0, 1);
    for m = 1:M
        idx = 8*(m-1) + (1:8);
        row = zeros(1, N); row(idx) = basis(0, 0, T(m)); A(end+1, :) = row; b(end+1) = W(m, ax);   %#ok<AGROW>
        row = zeros(1, N); row(idx) = basis(1, 0, T(m)); A(end+1, :) = row; b(end+1) = W(m+1, ax); %#ok<AGROW>
    end
    for k = 1:3                                            % rest at both ends
        row = zeros(1, N); row(1:8) = basis(0, k, T(1)); A(end+1, :) = row; b(end+1) = 0;              %#ok<AGROW>
        row = zeros(1, N); row(8*(M-1) + (1:8)) = basis(1, k, T(M)); A(end+1, :) = row; b(end+1) = 0;  %#ok<AGROW>
    end
    for m = 1:M-1                                          % vel, acc, jerk continuous
        for k = 1:3
            row = zeros(1, N);
            row(8*(m-1) + (1:8)) = basis(1, k, T(m));
            row(8*m + (1:8))     = -basis(0, k, T(m+1));
            A(end+1, :) = row; b(end+1) = 0;               %#ok<AGROW>
        end
    end
    K = [2*Q, A.'; A, zeros(size(A, 1))];
    sol = K \ [zeros(N, 1); b(:)];
    coeff(:, :, ax) = reshape(sol(1:N), 8, M);
end
end

function pk = legPeaks(coeff, T)
% Peak speed on each leg.
pk = zeros(1, numel(T));
for m = 1:numel(T)
    tau = linspace(0, 1, 120);
    v = zeros(numel(tau), 2);
    for ax = 1:2
        for n = 1:numel(tau)
            v(n, ax) = basis(tau(n), 1, T(m)) * coeff(:, m, ax);
        end
    end
    pk(m) = max(vecnorm(v, 2, 2));
end
end

function Qn = snapCost()
% int_0^1 (d4/dtau4 sum c_j tau^j)^2 dtau, powers j = 0..7
Qn = zeros(8);
for i = 4:7
    for j = 4:7
        Qn(i+1, j+1) = (factorial(i)/factorial(i-4)) * (factorial(j)/factorial(j-4)) / (i + j - 7);
    end
end
end

function row = basis(tau, k, Tm)
% k-th time derivative of [1 tau ... tau^7] at tau, in physical time units
row = zeros(1, 8);
for j = k:7
    row(j+1) = factorial(j) / factorial(j-k) * tau^(j-k);
end
row = row / Tm^k;
end

function [P, V] = sampleMinSnap(coeff, T, t)
edges = [0 cumsum(T)];
P = zeros(numel(t), 2);  V = zeros(numel(t), 2);
for n = 1:numel(t)
    m = min(find(t(n) >= edges, 1, 'last'), numel(T));
    tau = (t(n) - edges(m)) / T(m);
    for ax = 1:2
        c = coeff(:, m, ax);
        P(n, ax) = basis(tau, 0, T(m)) * c;
        V(n, ax) = basis(tau, 1, T(m)) * c;
    end
end
end
