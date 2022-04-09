//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = MVP*vec4(vp.x, vp.y, 0, 1);		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao, vbo;	   // virtual world on the GPU
const int nv = 100;

float random() { return (float)rand() / RAND_MAX; }

class Circle {
    vec2 position;
    float radius;
    vec3 color;
    vec2 perimeterPoints[nv];
public:
    Circle(vec2 pos, float r, vec3 col) {
        position = pos;
        radius = r;
        color = col;
        for (int i = 0; i < nv; i++) {
            float fi = i * 2  * M_PI / nv;
            perimeterPoints[i] = vec2(cosf(fi)*radius, sinf(fi)*radius);
        }
    }

    vec2* getPerimeterPoints() {
        return perimeterPoints;
    }
    void draw() {


        gpuProgram.setUniform(color, "color");
        glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
                     sizeof(vec2)*nv,  // # bytes
                     perimeterPoints,	      	// address
                     GL_STATIC_DRAW);	// we do not change later

        gpuProgram.Use();

        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");

        int mvpLocation = glGetUniformLocation(gpuProgram.getId(), "MVP");

        float matrix[] = {
                1,0,0,0,
                0, 1, 0, 0,
                0,0,1,0,
                position.x,position.y, 0,1
        };
        glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, matrix);
        glDrawArrays(GL_TRIANGLE_FAN, 0, nv);
    }
};

class Atom : Circle {
    float charge;
    float mass;
};

class Molecule {
    std::vector<Circle> atoms;
    // lines
    vec2 origin;
};

std::vector<Circle> circles;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "outColor");

    glGenVertexArrays(1, &vao);	// get 1 vao id
    glBindVertexArray(vao);		// make it active

    glGenBuffers(1, &vbo);	// Generate 1 buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glEnableVertexAttribArray(0);  // AttribArray 0
    glVertexAttribPointer(0,       // vbo -> AttribArray 0
                          2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
                          0, NULL); 		     // stride, offset: tightly packed


    // Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
    float vertices[] = { -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f };
//    circles.push_back(Circle(vec2(0.5f,0), 0.1f, vec3(1, 0, 0)));
//    circles.push_back(Circle(vec2(0.9f,0.3f), 0.4f, vec3(0, 1, 0)));
//    circles.push_back(Circle(vec2(-0.5f,-0.4f), 0.7f, vec3(0, 0, 1)));

    for (int i = 0; i < 1000; i++) {
        circles.push_back(Circle(vec2((random()*2)-1,(random()*2)-1), 1.0f*random()/20, vec3(random(), random(), random())));
    }

}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);     // background color
    glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

    for (auto c : circles){c.draw();}

//    // Set color to (0, 1, 0) = green
//    int location = glGetUniformLocation(gpuProgram.getId(), "color");
//    glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats
//
//    float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix,
//                              0, 1, 0, 0,    // row-major!
//                              0, 0, 1, 0,
//                              0, 0, 0, 1 };
//
//    location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
//    glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
//
//    glBindVertexArray(vao);  // Draw call
//    glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 /*# Elements*/);

    glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    // Convert to normalized device space
    float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
    float cY = 1.0f - 2.0f * pY / windowHeight;
    printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    // Convert to normalized device space
    float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
    float cY = 1.0f - 2.0f * pY / windowHeight;

    char * buttonStat;
    switch (state) {
        case GLUT_DOWN: buttonStat = "pressed"; break;
        case GLUT_UP:   buttonStat = "released"; break;
    }

    switch (button) {
        case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
        case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
        case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
    }
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}