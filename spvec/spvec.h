struct Point {double x,y;};

static double polygon_aos[1016] ={-6.72,49.975,-6.72,49.995,-6.695,50.045,-6.645,50.095,-6.61,50.12,-6.5,50.17,-6.37,50.2,-6.23,50.2,-6.115,50.17,-6.055,50.14,-6.05,50.17,-6.02,50.225,-5.96,50.285,-5.9,50.33,-5.695,50.415,-5.63,50.43,-5.555,50.435,-5.54,50.445,-5.5,50.45,-5.48,50.465,-5.435,50.525,-5.4,50.55,-5.395,50.585,-5.375,50.625,-5.395,51.23,-5.38,51.25,-5.11,51.345,-4.735,51.44,-4.615,51.425,-4.44,51.38,-3.96,51.42,-3.645,51.34,-3.31,51.305,-3.255,51.315,-3.195,51.37,-3.115,51.38,-3.005,51.475,-2.735,51.56,-2.715,51.6,-2.685,51.605,-2.68,51.62,-2.695,51.63,-2.705,51.655,-2.705,51.675,-2.695,51.685,-2.71,51.73,-2.705,51.745,-2.695,51.75,-2.7,51.76,-2.695,51.815,-2.755,51.82,-2.795,51.85,-2.8,51.87,-2.845,51.885,-2.85,51.895,-2.88,51.9,-2.88,51.91,-2.945,51.885,-2.98,51.885,-2.995,51.905,-3.025,51.915,-3.035,51.94,-3.045,51.94,-3.05,51.95,-3.085,51.97,-3.085,51.98,-3.12,52.02,-3.11,52.04,-3.125,52.045,-3.145,52.07,-3.14,52.1,-3.16,52.115,-3.165,52.13,-3.14,52.155,-3.14,52.18,-3.12,52.19,-3.12,52.215,-3.095,52.225,-3.09,52.25,-3.065,52.255,-3.065,52.265,-3.05,52.275,-3.035,52.275,-3.02,52.325,-3.075,52.33,-3.08,52.34,-3.105,52.345,-3.12,52.36,-3.15,52.37,-3.155,52.365,-3.175,52.375,-3.185,52.39,-3.24,52.41,-3.255,52.435,-3.25,52.465,-3.235,52.47,-3.225,52.485,-3.145,52.51,-3.155,52.52,-3.155,52.545,-3.145,52.555,-3.16,52.575,-3.16,52.59,-3.15,52.605,-3.125,52.605,-3.11,52.615,-3.11,52.625,-3.1,52.625,-3.1,52.655,-3.07,52.665,-3.065,52.7,-3.045,52.715,-3.035,52.735,-3.04,52.745,-3.085,52.75,-3.1,52.765,-3.17,52.775,-3.18,52.785,-3.18,52.795,-3.19,52.8,-3.19,52.825,-3.18,52.835,-3.185,52.855,-3.17,52.87,-3.175,52.88,-3.165,52.905,-3.15,52.915,-3.13,52.915,-3.125,52.935,-3.105,52.95,-3.08,52.945,-3.04,52.95,-3.02,52.975,-2.99,52.98,-2.98,52.99,-2.965,52.99,-2.935,52.96,-2.92,52.965,-2.915,52.96,-2.905,52.97,-2.84,52.965,-2.795,52.925,-2.75,52.945,-2.75,52.96,-2.775,52.975,-2.815,52.97,-2.85,52.98,-2.855,52.995,-2.875,53.005,-2.89,53.055,-2.9,53.07,-2.92,53.08,-2.92,53.095,-2.975,53.115,-2.985,53.13,-3.015,53.145,-3.005,53.175,-2.965,53.18,-2.955,53.185,-2.96,53.195,-3.03,53.23,-3.1,53.24,-3.175,53.295,-3.235,53.32,-3.31,53.385,-3.42,53.435,-3.555,53.52,-3.555,53.54,-3.54,53.55,-3.46,53.565,-3.45,53.62,-3.425,53.665,-3.395,53.695,-3.415,53.745,-3.415,53.865,-3.495,53.91,-3.58,53.99,-3.58,54,-3.615,54.04,-3.62,54.065,-3.745,54.18,-3.745,54.19,-3.775,54.23,-3.79,54.275,-3.8,54.275,-3.815,54.295,-3.825,54.295,-3.865,54.33,-3.905,54.35,-3.96,54.41,-4.085,54.48,-4.095,54.51,-4.07,54.545,-3.865,54.64,-3.65,54.795,-3.555,54.835,-3.48,54.925,-3.355,54.96,-3.325,54.98,-3.3,54.985,-3.195,54.985,-3.155,54.97,-3.125,54.995,-3.09,54.99,-3.07,55.005,-3.06,55.025,-3.075,55.04,-3.075,55.055,-3.055,55.075,-2.965,55.07,-2.955,55.085,-2.91,55.1,-2.88,55.125,-2.865,55.125,-2.84,55.155,-2.825,55.16,-2.81,55.155,-2.755,55.175,-2.735,55.19,-2.715,55.19,-2.685,55.235,-2.655,55.24,-2.665,55.25,-2.665,55.27,-2.615,55.305,-2.59,55.31,-2.57,55.335,-2.535,55.34,-2.51,55.365,-2.485,55.375,-2.41,55.38,-2.385,55.37,-2.36,55.385,-2.365,55.405,-2.35,55.425,-2.31,55.43,-2.265,55.455,-2.215,55.455,-2.21,55.46,-2.225,55.47,-2.225,55.485,-2.24,55.49,-2.25,55.505,-2.255,55.54,-2.28,55.55,-2.285,55.56,-2.3,55.56,-2.31,55.59,-2.355,55.625,-2.35,55.65,-2.33,55.665,-2.25,55.67,-2.23,55.695,-2.195,55.715,-2.195,55.73,-2.185,55.74,-2.17,55.74,-2.15,55.76,-2.13,55.76,-2.11,55.78,-2.1,55.81,-2.05,55.825,-1.975,55.865,-1.815,55.905,-1.735,55.935,-1.715,55.935,-1.67,55.895,-1.61,55.885,-1.565,55.865,-1.44,55.845,-1.395,55.83,-1.305,55.78,-1.26,55.735,-1.235,55.685,-1.23,55.63,-1.245,55.57,-1.235,55.565,-1.22,55.525,-1.205,55.435,-1.185,55.415,-1.165,55.37,-1.16,55.325,-1.17,55.29,-1.155,55.28,-1.14,55.25,-1.13,55.18,-1.11,55.16,-1.11,55.15,-1.07,55.115,-1.065,55.1,-1.04,55.085,-1,55.03,-0.985,54.985,-0.985,54.925,-0.97,54.91,-0.96,54.875,-0.89,54.84,-0.855,54.8,-0.825,54.8,-0.78,54.785,-0.61,54.755,-0.555,54.73,-0.51,54.72,-0.455,54.69,-0.39,54.675,-0.305,54.635,-0.195,54.55,-0.195,54.54,-0.135,54.49,-0.135,54.48,-0.1,54.445,-0.075,54.405,-0.04,54.395,0.015,54.365,0.05,54.33,0.14,54.3,0.205,54.26,0.255,54.21,0.275,54.175,0.285,54.145,0.285,54.09,0.255,54.02,0.2,53.965,0.31,53.87,0.32,53.87,0.345,53.84,0.38,53.82,0.4,53.795,0.415,53.79,0.46,53.735,0.49,53.68,0.505,53.63,0.505,53.54,0.53,53.52,0.55,53.48,0.575,53.46,0.585,53.435,0.61,53.41,0.61,53.4,0.655,53.35,0.665,53.32,0.68,53.305,0.71,53.21,0.78,53.2,1.035,53.2,1.205,53.165,1.275,53.165,1.415,53.145,1.53,53.11,1.66,53.055,1.675,53.04,1.695,53.035,1.7,53.025,1.82,52.97,1.95,52.885,2,52.84,2.02,52.805,2.03,52.8,2.09,52.675,2.095,52.655,2.09,52.56,2.11,52.52,2.105,52.43,2.085,52.39,2.075,52.385,2.07,52.35,2.05,52.31,2.025,52.285,2,52.23,1.97,52.2,1.96,52.13,1.93,52.08,1.905,52.005,1.85,51.945,1.81,51.925,1.805,51.915,1.72,51.885,1.675,51.845,1.625,51.82,1.62,51.795,1.595,51.75,1.545,51.7,1.46,51.655,1.445,51.64,1.39,51.62,1.535,51.605,1.62,51.58,1.71,51.53,1.75,51.49,1.79,51.41,1.79,51.345,1.775,51.295,1.76,51.275,1.895,51.22,1.91,51.2,1.9,51.18,1.735,51.08,1.555,51.02,1.365,50.93,1.3,50.885,1.28,50.81,1.24,50.77,1.185,50.735,1.12,50.71,1.005,50.69,0.865,50.695,0.84,50.68,0.735,50.65,0.555,50.625,0.5,50.58,0.43,50.545,0.3,50.515,0.185,50.515,0.115,50.525,-0.015,50.555,-0.05,50.57,-0.225,50.605,-0.36,50.585,-0.515,50.58,-0.54,50.575,-0.545,50.565,-0.615,50.53,-0.685,50.51,-0.78,50.5,-0.87,50.505,-0.93,50.45,-1.015,50.405,-1.145,50.37,-1.25,50.355,-1.345,50.355,-1.41,50.365,-1.5,50.395,-1.57,50.44,-1.66,50.445,-1.695,50.455,-1.7,50.445,-1.71,50.445,-1.75,50.415,-1.85,50.38,-2.05,50.355,-2.13,50.36,-2.195,50.375,-2.2,50.365,-2.265,50.33,-2.36,50.3,-2.525,50.295,-2.595,50.31,-2.67,50.34,-2.745,50.395,-2.775,50.435,-2.785,50.465,-2.86,50.495,-3.085,50.465,-3.14,50.435,-3.16,50.335,-3.235,50.225,-3.26,50.21,-3.275,50.19,-3.315,50.17,-3.345,50.115,-3.38,50.08,-3.47,50.03,-3.645,49.985,-3.795,49.985,-3.875,49.995,-3.98,50.02,-4.085,50.075,-4.16,50.09,-4.295,50.095,-4.385,50.12,-4.515,50.105,-4.53,50.085,-4.54,50.085,-4.55,50.07,-4.6,50.04,-4.65,50.02,-4.735,50.005,-4.77,49.935,-4.835,49.87,-4.885,49.835,-4.925,49.815,-4.94,49.815,-5.02,49.77,-5.14,49.74,-5.27,49.74,-5.365,49.76,-5.425,49.785,-5.5,49.835,-5.625,49.815,-5.725,49.815,-5.825,49.835,-5.905,49.875,-5.93,49.88,-5.955,49.835,-6,49.8,-6.03,49.765,-6.15,49.7,-6.28,49.66,-6.43,49.655,-6.52,49.675,-6.57,49.695,-6.61,49.725,-6.62,49.725,-6.67,49.775,-6.69,49.81,-6.705,49.875,-6.725,49.92,-6.72,49.975};

static double polygon_soa[1016] ={-6.72,-6.72,-6.695,-6.645,-6.61,-6.5,-6.37,-6.23,49.975,49.995,50.045,50.095,50.12,50.17,50.2,50.2,-6.115,-6.055,-6.05,-6.02,-5.96,-5.9,-5.695,-5.63,50.17,50.14,50.17,50.225,50.285,50.33,50.415,50.43,-5.555,-5.54,-5.5,-5.48,-5.435,-5.4,-5.395,-5.375,50.435,50.445,50.45,50.465,50.525,50.55,50.585,50.625,-5.395,-5.38,-5.11,-4.735,-4.615,-4.44,-3.96,-3.645,51.23,51.25,51.345,51.44,51.425,51.38,51.42,51.34,-3.31,-3.255,-3.195,-3.115,-3.005,-2.735,-2.715,-2.685,51.305,51.315,51.37,51.38,51.475,51.56,51.6,51.605,-2.68,-2.695,-2.705,-2.705,-2.695,-2.71,-2.705,-2.695,51.62,51.63,51.655,51.675,51.685,51.73,51.745,51.75,-2.7,-2.695,-2.755,-2.795,-2.8,-2.845,-2.85,-2.88,51.76,51.815,51.82,51.85,51.87,51.885,51.895,51.9,-2.88,-2.945,-2.98,-2.995,-3.025,-3.035,-3.045,-3.05,51.91,51.885,51.885,51.905,51.915,51.94,51.94,51.95,-3.085,-3.085,-3.12,-3.11,-3.125,-3.145,-3.14,-3.16,51.97,51.98,52.02,52.04,52.045,52.07,52.1,52.115,-3.165,-3.14,-3.14,-3.12,-3.12,-3.095,-3.09,-3.065,52.13,52.155,52.18,52.19,52.215,52.225,52.25,52.255,-3.065,-3.05,-3.035,-3.02,-3.075,-3.08,-3.105,-3.12,52.265,52.275,52.275,52.325,52.33,52.34,52.345,52.36,-3.15,-3.155,-3.175,-3.185,-3.24,-3.255,-3.25,-3.235,52.37,52.365,52.375,52.39,52.41,52.435,52.465,52.47,-3.225,-3.145,-3.155,-3.155,-3.145,-3.16,-3.16,-3.15,52.485,52.51,52.52,52.545,52.555,52.575,52.59,52.605,-3.125,-3.11,-3.11,-3.1,-3.1,-3.07,-3.065,-3.045,52.605,52.615,52.625,52.625,52.655,52.665,52.7,52.715,-3.035,-3.04,-3.085,-3.1,-3.17,-3.18,-3.18,-3.19,52.735,52.745,52.75,52.765,52.775,52.785,52.795,52.8,-3.19,-3.18,-3.185,-3.17,-3.175,-3.165,-3.15,-3.13,52.825,52.835,52.855,52.87,52.88,52.905,52.915,52.915,-3.125,-3.105,-3.08,-3.04,-3.02,-2.99,-2.98,-2.965,52.935,52.95,52.945,52.95,52.975,52.98,52.99,52.99,-2.935,-2.92,-2.915,-2.905,-2.84,-2.795,-2.75,-2.75,52.96,52.965,52.96,52.97,52.965,52.925,52.945,52.96,-2.775,-2.815,-2.85,-2.855,-2.875,-2.89,-2.9,-2.92,52.975,52.97,52.98,52.995,53.005,53.055,53.07,53.08,-2.92,-2.975,-2.985,-3.015,-3.005,-2.965,-2.955,-2.96,53.095,53.115,53.13,53.145,53.175,53.18,53.185,53.195,-3.03,-3.1,-3.175,-3.235,-3.31,-3.42,-3.555,-3.555,53.23,53.24,53.295,53.32,53.385,53.435,53.52,53.54,-3.54,-3.46,-3.45,-3.425,-3.395,-3.415,-3.415,-3.495,53.55,53.565,53.62,53.665,53.695,53.745,53.865,53.91,-3.58,-3.58,-3.615,-3.62,-3.745,-3.745,-3.775,-3.79,53.99,54,54.04,54.065,54.18,54.19,54.23,54.275,-3.8,-3.815,-3.825,-3.865,-3.905,-3.96,-4.085,-4.095,54.275,54.295,54.295,54.33,54.35,54.41,54.48,54.51,-4.07,-3.865,-3.65,-3.555,-3.48,-3.355,-3.325,-3.3,54.545,54.64,54.795,54.835,54.925,54.96,54.98,54.985,-3.195,-3.155,-3.125,-3.09,-3.07,-3.06,-3.075,-3.075,54.985,54.97,54.995,54.99,55.005,55.025,55.04,55.055,-3.055,-2.965,-2.955,-2.91,-2.88,-2.865,-2.84,-2.825,55.075,55.07,55.085,55.1,55.125,55.125,55.155,55.16,-2.81,-2.755,-2.735,-2.715,-2.685,-2.655,-2.665,-2.665,55.155,55.175,55.19,55.19,55.235,55.24,55.25,55.27,-2.615,-2.59,-2.57,-2.535,-2.51,-2.485,-2.41,-2.385,55.305,55.31,55.335,55.34,55.365,55.375,55.38,55.37,-2.36,-2.365,-2.35,-2.31,-2.265,-2.215,-2.21,-2.225,55.385,55.405,55.425,55.43,55.455,55.455,55.46,55.47,-2.225,-2.24,-2.25,-2.255,-2.28,-2.285,-2.3,-2.31,55.485,55.49,55.505,55.54,55.55,55.56,55.56,55.59,-2.355,-2.35,-2.33,-2.25,-2.23,-2.195,-2.195,-2.185,55.625,55.65,55.665,55.67,55.695,55.715,55.73,55.74,-2.17,-2.15,-2.13,-2.11,-2.1,-2.05,-1.975,-1.815,55.74,55.76,55.76,55.78,55.81,55.825,55.865,55.905,-1.735,-1.715,-1.67,-1.61,-1.565,-1.44,-1.395,-1.305,55.935,55.935,55.895,55.885,55.865,55.845,55.83,55.78,-1.26,-1.235,-1.23,-1.245,-1.235,-1.22,-1.205,-1.185,55.735,55.685,55.63,55.57,55.565,55.525,55.435,55.415,-1.165,-1.16,-1.17,-1.155,-1.14,-1.13,-1.11,-1.11,55.37,55.325,55.29,55.28,55.25,55.18,55.16,55.15,-1.07,-1.065,-1.04,-1,-0.985,-0.985,-0.97,-0.96,55.115,55.1,55.085,55.03,54.985,54.925,54.91,54.875,-0.89,-0.855,-0.825,-0.78,-0.61,-0.555,-0.51,-0.455,54.84,54.8,54.8,54.785,54.755,54.73,54.72,54.69,-0.39,-0.305,-0.195,-0.195,-0.135,-0.135,-0.1,-0.075,54.675,54.635,54.55,54.54,54.49,54.48,54.445,54.405,-0.04,0.015,0.05,0.14,0.205,0.255,0.275,0.285,54.395,54.365,54.33,54.3,54.26,54.21,54.175,54.145,0.285,0.255,0.2,0.31,0.32,0.345,0.38,0.4,54.09,54.02,53.965,53.87,53.87,53.84,53.82,53.795,0.415,0.46,0.49,0.505,0.505,0.53,0.55,0.575,53.79,53.735,53.68,53.63,53.54,53.52,53.48,53.46,0.585,0.61,0.61,0.655,0.665,0.68,0.71,0.78,53.435,53.41,53.4,53.35,53.32,53.305,53.21,53.2,1.035,1.205,1.275,1.415,1.53,1.66,1.675,1.695,53.2,53.165,53.165,53.145,53.11,53.055,53.04,53.035,1.7,1.82,1.95,2,2.02,2.03,2.09,2.095,53.025,52.97,52.885,52.84,52.805,52.8,52.675,52.655,2.09,2.11,2.105,2.085,2.075,2.07,2.05,2.025,52.56,52.52,52.43,52.39,52.385,52.35,52.31,52.285,2,1.97,1.96,1.93,1.905,1.85,1.81,1.805,52.23,52.2,52.13,52.08,52.005,51.945,51.925,51.915,1.72,1.675,1.625,1.62,1.595,1.545,1.46,1.445,51.885,51.845,51.82,51.795,51.75,51.7,51.655,51.64,1.39,1.535,1.62,1.71,1.75,1.79,1.79,1.775,51.62,51.605,51.58,51.53,51.49,51.41,51.345,51.295,1.76,1.895,1.91,1.9,1.735,1.555,1.365,1.3,51.275,51.22,51.2,51.18,51.08,51.02,50.93,50.885,1.28,1.24,1.185,1.12,1.005,0.865,0.84,0.735,50.81,50.77,50.735,50.71,50.69,50.695,50.68,50.65,0.555,0.5,0.43,0.3,0.185,0.115,-0.015,-0.05,50.625,50.58,50.545,50.515,50.515,50.525,50.555,50.57,-0.225,-0.36,-0.515,-0.54,-0.545,-0.615,-0.685,-0.78,50.605,50.585,50.58,50.575,50.565,50.53,50.51,50.5,-0.87,-0.93,-1.015,-1.145,-1.25,-1.345,-1.41,-1.5,50.505,50.45,50.405,50.37,50.355,50.355,50.365,50.395,-1.57,-1.66,-1.695,-1.7,-1.71,-1.75,-1.85,-2.05,50.44,50.445,50.455,50.445,50.445,50.415,50.38,50.355,-2.13,-2.195,-2.2,-2.265,-2.36,-2.525,-2.595,-2.67,50.36,50.375,50.365,50.33,50.3,50.295,50.31,50.34,-2.745,-2.775,-2.785,-2.86,-3.085,-3.14,-3.16,-3.235,50.395,50.435,50.465,50.495,50.465,50.435,50.335,50.225,-3.26,-3.275,-3.315,-3.345,-3.38,-3.47,-3.645,-3.795,50.21,50.19,50.17,50.115,50.08,50.03,49.985,49.985,-3.875,-3.98,-4.085,-4.16,-4.295,-4.385,-4.515,-4.53,49.995,50.02,50.075,50.09,50.095,50.12,50.105,50.085,-4.54,-4.55,-4.6,-4.65,-4.735,-4.77,-4.835,-4.885,50.085,50.07,50.04,50.02,50.005,49.935,49.87,49.835,-4.925,-4.94,-5.02,-5.14,-5.27,-5.365,-5.425,-5.5,49.815,49.815,49.77,49.74,49.74,49.76,49.785,49.835,-5.625,-5.725,-5.825,-5.905,-5.93,-5.955,-6,-6.03,49.815,49.815,49.835,49.875,49.88,49.835,49.8,49.765,-6.15,-6.28,-6.43,-6.52,-6.57,-6.61,-6.62,-6.67,49.7,49.66,49.655,49.675,49.695,49.725,49.725,49.775,-6.69,-6.705,-6.725,-6.72,49.81,49.875,49.92,49.975};
