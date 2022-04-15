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

GPUProgram gpuProgram;
const int nv = 100;

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
            gpuProgram.setUniform(M(), "MVP");
            glBindVertexArray(vao);
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
        phi = t;
    }

    void translate(vec2 m) {
        wTranslate.x = m.x;
        wTranslate.y = m.y;
        // TODO translate to +=
    }
};

class Atom : Object {
public:
    float chargePowerMinus19C = 1.602f;
    float massPowerMinus24Kg = 1.674f;
    vec2 position;
    float radius;
    vec3 color;

    Atom(vec2 pos, int chargeMultiplier) {
        position = pos;
        chargePowerMinus19C *= chargeMultiplier;
        massPowerMinus24Kg *= ((rand() % 100) + 100);
        radius = massPowerMinus24Kg / 5000;
        if(chargePowerMinus19C < 0)
            color = vec3(0, 0, chargePowerMinus19C*(-1) / 1000);
        else
            color = vec3(chargePowerMinus19C / 1000, 0, 0);


        for (int i = 0; i < nv; i++) {
            float fi = i * 2  * M_PI / nv;
            vtx.push_back(vec2(cosf(fi)*radius, sinf(fi)*radius) + position);
        }
    }
    void drawAtom() {
        Draw(GL_TRIANGLE_FAN, this->color);
    }

    float getMass() const {
        return massPowerMinus24Kg;
    }

    float getX() const {
        return position.x;
    }

    void setVertices(vec2 vec) {
        for (auto &v : vtx) {
            v.x -= vec.x;
            v.y -= vec.y;
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
    }
};

class Bond : Object {
public:
    vec2 pointFrom;
    vec2 pointTo;
    vec2 normalized;
    float distanceUnit;
    float length;

    Bond(vec2 pFrom, vec2 pTo) {
        pointFrom = pFrom;
        pointTo = pTo;
        length = sqrt((pointFrom.x-pointTo.x)*(pointFrom.x-pointTo.x)+(pointFrom.y-pointTo.y)*(pointFrom.y-pointTo.y));
        distanceUnit = length / nv;
        normalized = (pointTo - pointFrom) / length;
        for (int i = 0; i < nv; i++) {
            vtx.push_back(pointFrom + (normalized * distanceUnit * i));
        }
    }

    void drawBond() {
        Draw(GL_LINE_STRIP, vec3(1,1,1));
    }

    void setVertices(vec2 vec) {
        for (auto &v : vtx) {
            v.x -= vec.x;
            v.y -= vec.y;
        }
    }

    void rotateBond(float d) {
        phi = d;
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
        for (int i = 0; i < vtx.size() - 1; i++) {
            charges.push_back((rand() % 2000) - 1000);
            sum+=charges[i];
        }
        charges.push_back(sum * (-1));
        for (int i = 0; i < vtx.size(); i++) {
            atoms.push_back(Atom(vtx[i],charges[i]));
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
            v.x -= centerOfMass.x;
            v.y -= centerOfMass.y;
        }
        for (auto &a : atoms) {
            a.setVertices(centerOfMass);
        }

        for (auto &b : bonds) {
            b.setVertices(centerOfMass);
        }
    }

    void simulate(float sec) {
        vec2 r;
        vec2 v;
        float omega;
        float alpha;
        vec2 F;
        float M;
        std::vector<Atom> otherAtoms;

        for (auto &a : atoms) {
            for(auto &a : otherAtoms) {
                // F = // coulomb and medium
                // M = // 
            }
            // v += F / a.getMass() * sec;
            // r += v * sec;
            // omega += M / theta * sec;
            // alpha += omega * sec;
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
    }

    void draw() {
        if(notFirst) {
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
    }

    void simulate(float sec) {
        for (auto &m : molecules) {
            m.rotateMolecule(sec);
        }
        molecules[0].translateMolecule(vec2(0.5f, 0.5f));
        molecules[1].translateMolecule(vec2(-0.5f, -0.5f));
    //    molecules[0].simulate(sec);
    //    molecules[1].simulate(sec);
    }
};

Scene s;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glLineWidth(2);
    glPointSize(10);

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
    if (key == 32) {

        s.pressSpace();
        glutPostRedisplay();
    }
}


void onKeyboardUp(unsigned char key, int pX, int pY) {
}


void onMouseMotion(int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME);
    float sec = time / 1000.0f;
    s.simulate(sec);
    glutPostRedisplay();
}