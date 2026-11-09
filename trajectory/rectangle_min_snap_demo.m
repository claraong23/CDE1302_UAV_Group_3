%% rectangle_min_snap_demo
% Minimum-snap rectangle over the spots to cover, flown from the start.
%
% Frame: metres from the start point (0, 0); x = forward, y = right. The drone
% keeps its takeoff heading (nose +x) for the whole flight, so the 60 deg
% forward camera looks ahead and across the rectangle while it flies the edges.
%
% Route: start at rest at (0, 0) -> diagonal to the near-left corner ->
% clockwise round the rectangle (forward up the left edge, across the far
% edge, back down the right edge, across the near edge) -> diagonal home to
% (0, 0) at rest. Corners are rounded (corner_radius).
%
% The rectangle is the spots' rectangle shrunk where the camera still covers
% them: the far edge and the sides come in (the camera sees ahead and to the
% side), the near edge stays (moving it in would leave those spots behind the
% camera). With far_in 0 / left_in 0.2 / right_in -0.2 the corner spots stay fully in view for
% >= 5 s at 1.0 m height, the middle spot (3.25, -0.10) only ~1.6 s
% (checked with search_patterns.camera_coverage).
%
% Seventh-order minimum-snap polynomial per segment, C3 continuous. Every
% join between an edge and a corner is a straight-flight state (cruise speed
% along the edge, zero acceleration and jerk), so the edges are exactly
% straight at a steady speed and each corner is the minimum-snap turn between
% them. (Leaving acc/jerk free makes the optimiser bend the edges to build up
% the turn.) Each corner's duration is picked so its speed stays near cruise.
%
% Outputs beside this file:
%   rectangle_min_snap_traj.csv   t,x,y,z (z = 0: holds the takeoff height)
%   rectangle_min_snap.png        top-down path + candidate spots, and speed
%
% Fly it (no --traj-rotate; it already points +x):
%   python laptop/tag_mission.py --drone-id 3 --trajectory trajectory/rectangle_min_snap_traj.csv --confirm

clc; close all;
traj_dir = fileparts(mfilename('fullpath'));

%% User parameters
% Spots to fly over / cover (cm -> m), x forward, y right: the rectangle corners.
cover = [140 -140;
         410 -140;
         410   93;
         140   93] / 100;
near_in = 0.00;         % m the near edge moves forward (keep 0: camera looks ahead)
far_in  = 0.00;         % m the far edge moves back (negative = further out)
left_in  = 0.20;        % m the left edge moves in (right, towards the middle)
right_in = -0.20;       % m the right edge moves in (negative = further out to the right)
corner_radius = 0.30;   % m; how far before each corner the turn starts
v_limit       = 0.297;  % peak speed (m/s); tag_mission.py refuses > 0.30
ramp_factor   = 2.0;    % start/finish segments take ramp_factor * L / v
dt            = 0.01;   % output sample time (s); the uploader resamples

% Other likely target spots inside the rectangle (plot only).
candidates = [274   91;
              236  -62;
              325  -10] / 100;

x_near  = min(cover(:,1)) + near_in;
x_far   = max(cover(:,1)) - far_in;
y_left  = min(cover(:,2)) + left_in;
y_right = max(cover(:,2)) - right_in;

%% Corner polyline: start -> near-left corner -> clockwise -> home
C = [0      0;
     x_near y_left;
     x_far  y_left;
     x_far  y_right;
     x_near y_right;
     0      0];

%% Waypoints: each corner is replaced by a turn-in and a turn-out point
r = corner_radius;
W = C(1, :);  D = [0 0];  seg_len = [];  seg_kind = {};
for i = 2:size(C, 1) - 1
    din  = (C(i, :) - C(i-1, :)) / norm(C(i, :) - C(i-1, :));
    dout = (C(i+1, :) - C(i, :)) / norm(C(i+1, :) - C(i, :));
    assert(norm(C(i, :) - C(i-1, :)) > 2*r && norm(C(i+1, :) - C(i, :)) > 2*r - 1e-9, ...
        'corner_radius too big for the edge lengths');
    theta = acos(max(-1, min(1, dot(din, dout))));          % turn angle
    W(end+1, :) = C(i, :) - r*din;   D(end+1, :) = din;     %#ok<SAGROW>
    W(end+1, :) = C(i, :) + r*dout;  D(end+1, :) = dout;    %#ok<SAGROW>
    seg_kind{end+1} = 'straight';                           %#ok<SAGROW>
    seg_kind{end+1} = 'corner';                             %#ok<SAGROW>
    seg_len(end+1) = 0;                                     %#ok<SAGROW>
    seg_len(end+1) = r / tan(theta/2) * theta;              %#ok<SAGROW> % arc length
end
W(end+1, :) = C(end, :);  D(end+1, :) = [0 0];
seg_kind{end+1} = 'straight';
seg_len(end+1) = 0;
M = size(W, 1) - 1;                                         % number of segments
for m = 1:M
    if strcmp(seg_kind{m}, 'straight')
        seg_len(m) = norm(W(m+1, :) - W(m, :));
    end
end
seg_kind{1} = 'ramp';  seg_kind{M} = 'ramp';                % from / to rest

%% Corner timing: duration = factor * arc length / v; pick the factor that
% keeps each corner's speed closest to cruise (shape only depends on factor).
factors = 0.6:0.01:2.0;
corner_factor = ones(1, M);
for m = find(strcmp(seg_kind, 'corner'))
    best = inf;
    for f = factors
        c1 = solveMinSnap(W(m:m+1, :), D(m:m+1, :), f * seg_len(m));
        tt = linspace(0, f * seg_len(m), 200).';
        [~, Vc] = sampleMinSnap(c1, f * seg_len(m), tt);
        dev = max(abs(vecnorm(Vc, 2, 2) - 1));
        if dev < best, best = dev; corner_factor(m) = f; end
    end
end

%% Solve at a trial cruise speed, then rescale so the peak is v_limit
% (velocities scale with v and times with 1/v, so the shape is unchanged).
v = v_limit;
for pass = 1:2
    T = corner_factor .* seg_len / v;
    T(strcmp(seg_kind, 'ramp')) = ramp_factor * seg_len(strcmp(seg_kind, 'ramp')) / v;
    coeff = solveMinSnap(W, D * v, T);
    t = (0:dt:sum(T)).';
    if t(end) < sum(T), t(end+1) = sum(T); end              %#ok<AGROW>
    [P, V] = sampleMinSnap(coeff, T, t);
    speed = vecnorm(V, 2, 2);
    v = v * v_limit / max(speed);
end
P(end, :) = W(end, :);                                      % exact home point

fprintf('Rectangle x %.2f..%.2f, y %.2f..%.2f, corner radius %.2f m\n', ...
    x_near, x_far, y_left, y_right, r);
fprintf('Path spans x %.2f..%.2f, y %.2f..%.2f\n', min(P(:,1)), max(P(:,1)), min(P(:,2)), max(P(:,2)));
fprintf('Duration %.1f s; cruise %.3f m/s; peak %.3f m/s; mean %.3f m/s\n', ...
    t(end), v, max(speed), mean(speed));
assert(max(speed) <= v_limit + 1e-6, 'peak speed above v_limit');

%% Write CSV (t, x, y, z)
csv_path = fullfile(traj_dir, 'rectangle_min_snap_traj.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, '# Minimum-snap rectangle from rectangle_min_snap_demo.m (x forward, y right, from the start)\n');
fprintf(fid, '# rectangle x %.2f..%.2f, y %.2f..%.2f, corner radius %.2f m; fly WITHOUT --traj-rotate\n', ...
    x_near, x_far, y_left, y_right, r);
fprintf(fid, '# %.1f s, peak %.3f m/s, dt %.3f s\n', t(end), max(speed), dt);
fprintf(fid, 't,x,y,z\n');
fprintf(fid, '%.3f,%.4f,%.4f,0\n', [t P].');
fclose(fid);
fprintf('Wrote %s\n', csv_path);

%% Plot: top view (x forward = up, y right = right) and speed
fig = figure('Name', 'Rectangle min-snap', 'Color', 'w', 'Position', [100 100 1100 620]);
try, theme(fig, 'light'); catch, end                        % R2025 dark theme off
subplot(1, 2, 1);
plot(cover([1 2 3 4 1],2), cover([1 2 3 4 1],1), ':', ...
     'Color', [0.6 0.6 0.6]); hold on;
plot(P(:,2), P(:,1), 'Color', [0.08 0.45 0.78], 'LineWidth', 2.5);
plot(cover(:,2), cover(:,1), 'rx', 'MarkerSize', 12, 'LineWidth', 2.5);
plot(candidates(:,2), candidates(:,1), '+', 'Color', [0.9 0.5 0.1], 'MarkerSize', 10, 'LineWidth', 2);
spots = [cover; candidates];
for k = 1:size(spots, 1)
    text(spots(k,2) + 0.08, spots(k,1) + 0.08, ...
        sprintf('(%.2f, %.2f)', spots(k,1), spots(k,2)), 'Color', [0.7 0.1 0.1], 'FontSize', 8);
end
plot(0, 0, 'o', 'MarkerSize', 10, 'MarkerFaceColor', [0.95 0.55 0.1], 'Color', [0.95 0.55 0.1]);
step = max(1, round(numel(t) / 30));
quiver(P(1:step:end-1,2), P(1:step:end-1,1), V(1:step:end-1,2), V(1:step:end-1,1), 0.6, ...
       'Color', [0.08 0.45 0.78]);
axis equal; grid on; box on;
xlabel('y right (m)'); ylabel('x forward (m)');
title(sprintf('Min-snap rectangle, %.0f s (x = spots to cover, dotted = their rectangle)', t(end)));
subplot(1, 2, 2);
plot(t, speed, 'LineWidth', 1.5); grid on;
xlabel('time (s)'); ylabel('speed (m/s)');
title(sprintf('peak %.3f m/s', max(speed)));
png_path = fullfile(traj_dir, 'rectangle_min_snap.png');
exportgraphics(fig, png_path, 'Resolution', 130);
fprintf('Wrote %s\n', png_path);

%% Local functions
function coeff = solveMinSnap(W, Vw, T)
% Per axis: c (8 per segment, normalised time tau in [0, 1]) minimising
% sum_m T_m^-7 * int (d4p/dtau4)^2, subject to
%   positions at both ends of every segment,
%   velocity Vw at every waypoint (both sides), zero acc/jerk at start and end,
%   acceleration and jerk continuous at every interior waypoint.
M = numel(T);  N = 8 * M;
Q = zeros(N);
for m = 1:M
    idx = 8*(m-1) + (1:8);
    Q(idx, idx) = snapCost() / T(m)^7;
end
coeff = zeros(8, M, 2);
for ax = 1:2
    A = [];  b = [];
    for m = 1:M
        idx = 8*(m-1) + (1:8);
        row = zeros(1, N); row(idx) = basis(0, 0, T(m)); A = [A; row]; b = [b; W(m,   ax)];   %#ok<AGROW>
        row = zeros(1, N); row(idx) = basis(1, 0, T(m)); A = [A; row]; b = [b; W(m+1, ax)];   %#ok<AGROW>
        row = zeros(1, N); row(idx) = basis(0, 1, T(m)); A = [A; row]; b = [b; Vw(m,   ax)];  %#ok<AGROW>
        row = zeros(1, N); row(idx) = basis(1, 1, T(m)); A = [A; row]; b = [b; Vw(m+1, ax)];  %#ok<AGROW>
    end
    for k = 2:3                                         % rest at both ends
        row = zeros(1, N); row(1:8) = basis(0, k, T(1)); A = [A; row]; b = [b; 0];             %#ok<AGROW>
        row = zeros(1, N); row(8*(M-1) + (1:8)) = basis(1, k, T(M)); A = [A; row]; b = [b; 0]; %#ok<AGROW>
    end
    for m = 1:M-1                                       % joins: straight flight
        for k = 2:3                                     % (zero acc + jerk both sides,
            row = zeros(1, N); row(8*(m-1) + (1:8)) = basis(1, k, T(m));   % so C3 holds
            A = [A; row]; b = [b; 0];                   %#ok<AGROW>      % and the edges
            row = zeros(1, N); row(8*m + (1:8)) = basis(0, k, T(m+1));     % stay straight)
            A = [A; row]; b = [b; 0];                   %#ok<AGROW>
        end
    end
    if size(A, 1) == N                  % fully fixed by the boundary conditions:
        sol = A \ b;                     % the unique (so minimum-snap) polynomial
    else
        K = [2*Q, A.'; A, zeros(size(A, 1))];
        sol = K \ [zeros(N, 1); b];
    end
    coeff(:, :, ax) = reshape(sol(1:N), 8, M);
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
