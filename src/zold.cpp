// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Horvath Benedek
// Neptun : D86EP7
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
		gl_Position = vec4(vp.x, vp.y, 0, 1)*MVP;		// transform vp from modeling space to normalized device space
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

class Camera2D {
    vec2 wCenter; // center in world coordinates
    vec2 wSize;   // width and height in world coordinates
public:
    Camera2D() : wCenter(0, 0), wSize(2, 2) {}

    mat4 V() { return TranslateMatrix(-wCenter); }

    mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

    void Zoom(float s) { wSize = wSize * s; }

    void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;        // 2D camera

GPUProgram gpuProgram;
const int nv = 100;
const float epsVacuum10Emin12 = 8.85;
const float epsWater10Emin12 = 711.54;

float randomFloat() { return (float)rand() / RAND_MAX; }

int getIndex(std::vector<vec2> &v, vec2 &item) {
    for (int i = 0; i < v.size(); ++i) {
        if (item.x == v[i].x && item.y == v[i].y) {
            return i;
        }
    }
}

class Object {
public:
    unsigned int vao, vbo;
    std::vector<vec2> vtx;
    float sx = 1;
    float sy = 1;
    vec2 wTranslate = vec2(0,0);
    float phi = 0;
    Object() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    mat4 M() {
        mat4 Mscale(sx, 0, 0, 0,
                    0, sy, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 1);

        mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
                     -sinf(phi), cosf(phi), 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1);

        mat4 Mtranslate(1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 0, 0,
                        wTranslate.x, wTranslate.y, 0, 1);

        return Mscale * Mrotate * Mtranslate;
    }

    void Draw(int type, vec3 color) {
        int count = vtx.size();
        if (count > 0) {
            updateGPU();
            gpuProgram.setUniform(color, "color");
            mat4 MVP = M() * camera.V() * camera.P();
            gpuProgram.setUniform(MVP, "MVP");
            glDrawArrays(type, 0, count);
        }
    }

    void updateGPU() {
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(vec2),
                     &vtx[0], GL_DYNAMIC_DRAW);
    }

    void rotate(float t) {
        phi += t;
    }

    void translate(vec2 m) {
        wTranslate = wTranslate + m;
    }
};

class Atom : Object {
public:
    float chargePowerMinus19C = 1.602f;
    float massPowerMinus24Kg = 1.674f;

    vec2 position;
    vec2 speed = 0;
    vec2 force = 0;

    float radius;
    vec3 color;

    Atom(vec2 pos, int chargeMultiplier) {
        position = pos;
        chargePowerMinus19C *= chargeMultiplier;
        massPowerMinus24Kg *= ((rand() % 100) + 1);
        radius = massPowerMinus24Kg / 5000 + 0.05f;
        // TODO itt kene valami
        if(chargePowerMinus19C < 0)
            color = vec3(0, 0, chargePowerMinus19C*(-1) / 500);
        else
            color = vec3(chargePowerMinus19C / 500, 0, 0);


        for (int i = 0; i < nv; i++) {
            float fi = i * 2  * M_PI / nv;
            vtx.push_back(vec2(cosf(fi)*radius, sinf(fi)*radius) + position);
        }
    }
    void drawAtom() {
        Draw(GL_TRIANGLE_FAN, this->color);
        Draw(GL_POINT, this->color);
    }

    float getMass() const {
        return massPowerMinus24Kg;
    }

    float getX() const {
        return position.x;
    }

    void setVertices(vec2 vec) {
        position = position + vec;
        for (auto &v : vtx) {
            v = v + vec;
        }
    }

    float getY() const {
        return position.y;
    }

    void rotateAtom(float t) {
        rotate(t);
    }

    void translateAtom(vec2 v) {
        translate(v);
        position = position + v;
    }

    void setPosition(vec2 pos) {
        position = pos;
    }

    vec2 getPosition() {
        return position;
    }

    float getCharge() {
        return chargePowerMinus19C;
    }

    float getRadius() {
        return radius;
    }
};

class Bond : Object {
public:
    vec2 pointFrom;
    vec2 pointTo;
    vec2 normalized;
    float distanceUnit;
    float distance;

    Bond(vec2 pFrom, vec2 pTo) {
        pointFrom = pFrom;
        pointTo = pTo;
        //length = sqrt((pointFrom.x-pointTo.x)*(pointFrom.x-pointTo.x)+(pointFrom.y-pointTo.y)*(pointFrom.y-pointTo.y));
        distance = length(pointTo - pointFrom);
        distanceUnit = distance / nv;
        normalized = (pointTo - pointFrom) / distance;
        for (int i = 0; i < nv; i++) {
            vtx.push_back(pointFrom + (normalized * distanceUnit * i));
        }
    }

    void drawBond() {
        Draw(GL_LINE_STRIP, vec3(1,1,1));
    }

    void setVertices(vec2 vec) {
        for (auto &v : vtx) {
            v = v + vec;
        }
    }

    void rotateBond(float d) {
        rotate(d);
    }

    void translateBond(vec2 v) {
        translate(v);
    }
};

class Molecule : Object {
public:
    std::vector<Atom> atoms;
    std::vector<Bond> bonds;

    vec2 genRange;

    vec2 centerOfMass;
    float molMass;

    Molecule(vec2 range) {
        genRange = range;
    }

    void drawMolecule(vec3 color) {
        for (auto &b : bonds) {
            b.drawBond();
        }
        for (auto &a : atoms) {
            a.drawAtom();
        }
    }

    void addAtoms() {
        std::vector<int> charges;
        int sum = 0;
        // TODO test
        for (int i = 0; i < vtx.size() - 1; i++) {
            charges.push_back((rand() % 1000) - 500);
            sum+=charges[i];
        }
        // TODO test
        charges.push_back(sum * (-1));
        for (int i = 0; i < vtx.size(); i++) {
            atoms.push_back(Atom(vtx[i],charges[i]));
        }
        for (auto &a : atoms) {
            molMass += a.getMass();
        }
    }

    void addBonds() {

        std::vector<vec2> tree;
        std::vector<vec2> freeAtoms = std::vector<vec2>(vtx);
        vec2 start = vec2(freeAtoms[0]);

        tree.push_back(start);
        freeAtoms.erase(freeAtoms.begin() + getIndex(freeAtoms, start));

        while(tree.size() < vtx.size()) {
            float minDistance = 999;
            float distance;
            vec2 minPointTo;
            vec2 minPointFrom;
            for (auto &bound : tree) {
                for (auto &free : freeAtoms) {
                    distance = length(free - bound);
                    if (distance < minDistance) {
                        minDistance = distance;
                        minPointTo = vec2(free);
                        minPointFrom = vec2(bound);
                    }
                }
            }
            tree.push_back(minPointTo);
            bonds.push_back(Bond(minPointFrom, minPointTo));
            freeAtoms.erase(freeAtoms.begin() + getIndex(freeAtoms, minPointTo));
        }
    }

    void generateMolecule() {
        auto atomCount = rand() % 6 + 2;
        // TODO test
        //atomCount = 1;
        for (int i = 0; i < atomCount; i++) {
            vtx.push_back(
                    vec2((randomFloat() * fabs(genRange.x - genRange.y)) + genRange.x, randomFloat() * fabs(genRange.x - genRange.y)) + genRange.x
                    );
        }
        addAtoms();
        addBonds();
        calculateCenterOfMass();

        normalizeMolecule();
    }

    void rotateMolecule(float t) {
        rotate(t);
        for (auto &a : atoms) {
            a.rotateAtom(t);
        }
        for (auto &b : bonds) {
            b.rotateBond(t);
        }
    }

    void translateMolecule(vec2 v) {
        translate(v);
        for (auto &a : atoms) {
            a.translateAtom(v);
        }
        for (auto &b : bonds) {
            b.translateBond(v);
        }
        centerOfMass = centerOfMass + v;
    }

    void calculateCenterOfMass() {
        float sumImpX = 0;
        float sumImpY = 0;
        float sumMass = 0;
        for (const auto &a : atoms) {
            sumImpX += a.getMass() * a.getX();
            sumImpY += a.getMass() * a.getY();
            sumMass += a.getMass();
        }
        centerOfMass.x = sumImpX / sumMass;
        centerOfMass.y = sumImpY / sumMass;
    }

    void normalizeMolecule() {
        for (auto &v : vtx) {
            v = v - centerOfMass;
        }
        for (auto &a : atoms) {
            a.setVertices(-centerOfMass);
        }

        for (auto &b : bonds) {
            b.setVertices(-centerOfMass);
        }

        centerOfMass = centerOfMass - centerOfMass;
    }



    void interact(Molecule &other, int count) {
        float alpha = 0;
        vec2 acceleration = vec2(0,0);
        float omega = 0;
        vec2 speed = vec2(0,0);
        float dt = 0.01;

        for (int i = 0; i < count; i++) {
        vec3 torque = vec3(0,0,0);
        vec3 torqueSum = vec3(0,0,0);
        float theta = 0;

        vec2 FRotate = vec2(0, 0);
        vec2 FTranslate = vec2(0, 0);
            for (auto &a: atoms) {
                vec2 coulomb;
                vec2 coulombSum = vec2(0, 0);

                vec2 r = a.getPosition() - centerOfMass;
                float rLen = length(r);
                float aCharge = a.getCharge();
                vec2 aPos = a.getPosition();
                for (auto &o: other.atoms) {
                    float oCharge = o.getCharge();
                    vec2 oPos = o.getPosition();
                    vec2 aoVec = aPos - oPos;
                    coulomb = (10*(aCharge * oCharge * normalize(aoVec))) /
                            (2 * M_PI * epsVacuum10Emin12 * length(aoVec));
                    if(length(coulombSum) < 10000000) {
                        coulombSum = coulombSum + coulomb;
                    }
            //printf("col: %.2f\n", length(coulomb));
                }
                // TODO test
                FTranslate = FTranslate + coulombSum * normalize(r);
                FRotate = FRotate + (coulombSum - FTranslate);
                theta += a.getMass() * rLen * rLen * 4;
                torque = cross(r, FRotate);
                torqueSum = torqueSum + torque;
            }

            // TODO szim javit, hiperbola, kamera miert rossz/????

            //printf("Ft: %.2f\n", length(FTranslate));
            //printf("cs: %8.2f\n", length(FTranslate));
            acceleration = FTranslate / (molMass); // ok
            //printf("acc: %f\n", acceleration.x);
            speed = acceleration * dt; // ok
            speed = speed * 0.2;

            alpha = length((torqueSum) / theta);

            omega = alpha * dt; // ok
            omega = omega * 0.2;
            if (torqueSum.z < 0) omega *= -1;

            float angle = omega * dt;
            vec2 translateV = speed * dt; // ok
            // TODO test
            rotateMolecule(angle);
            translateMolecule(translateV);
            //printf("speed x: %f3\n", translateV.x);
        }
    }
};

class Scene {

    std::vector<Molecule> molecules;
    bool notFirst = false;
public:

    void init() {
        molecules.push_back(Molecule(vec2(-0.5f, 0.5f)));
        molecules.push_back(Molecule(vec2(-0.5f, 0.5f)));

        for (auto & m : molecules) {
            m.generateMolecule();
        }
        vec2 translate0 = vec2(-0.5, 0);
        vec2 translate1 = vec2(0.5, 0);
        molecules[0].translateMolecule(translate0);
        molecules[1].translateMolecule(translate1);
    }

    void draw() {
        if(true) {
            for (int i = 0; i < molecules.size(); i++) {
                molecules[i].drawMolecule(vec3(1, 1, 1));
            }
        }
    }

    void pressSpace() {
        notFirst = true;

        for (auto &m : molecules) {
            m = (Molecule(vec2(-0.5f, 0.5f)));
            m.generateMolecule();
        }
        vec2 translate0 = vec2(-0.5f, 0.1f);
        vec2 translate1 = vec2(0.5f, -0.09f);
        molecules[0].translateMolecule(translate0);
        molecules[1].translateMolecule(translate1);
    }

    void simulate(int count) {
        molecules[0].interact(molecules[1], count);
        molecules[1].interact(molecules[0], count);
    }
};

Scene s;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glLineWidth(2);
    glPointSize(50);

    srand(time(nullptr));

    s.init();

    gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
    glClearColor(0.3f, 0.3f, 0.3f, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    s.draw();

    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {

    switch (key) {
        case 32 :
            s.pressSpace();
            break;
//        case 's':
//            camera.Pan(vec2(-0.01, 0));
//            break;
//        case 'd':
//            camera.Pan(vec2(+0.01, 0));
//            break;
//        case 'e':
//            camera.Pan(vec2(0, 0.01));
//            break;
//        case 'x':
//            camera.Pan(vec2(0, -0.01));
//            break;
        case 'z':
            camera.Zoom(0.99f);
            break;
        case 'Z':
            camera.Zoom(1.01f);
            break;
    }
    glutPostRedisplay();
}


void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

long prevTime = 0;
int elapsed = 0;
int cycles = 0;
int deltat = 10; // ms

void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME);
    cycles = 0;
    elapsed += (time - prevTime);
    while (elapsed > deltat) {
        elapsed -= deltat;
        cycles++;
    }
    s.simulate(cycles);
    prevTime = time;
    glutPostRedisplay();
}