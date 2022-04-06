#include "framework.h"

// 3. buffer keszitese: buffer id letrehozasa
unsigned int buffer;
// 10. vertex array
unsigned int vertexArray;

// 6. adatok letrehozasa haromszoghoz
struct Vertex {
    vec2 position;
    vec3 color;
};// 16. 32:00


void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight); // 1. teljes kepernyot szeretnenk hasznalni

    // 7. adding vertices for triangle
    // important: it should be given counterclockwise, only one side is drawn
    Vertex data[] = {
            Vertex{vec2(0, 2 * 0.866f / 3), vec3(1,0,0)},
            Vertex{vec2(-0.5f, -0.866f / 3), vec3(0,1,0)},
            Vertex{vec2(0.5f, 0.866f / 3), vec3(0,0,1)}
    };

    // 11. vertex arrays, generate, bind
    // better before buffers
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    glGenBuffers(1, &buffer); // 4. hany buffer es tomb ahova meg akarjuk adni, itt "1 elemu tomb"
    glBindBuffer(GL_ARRAY_BUFFER, buffer); // 5. milyen fajtakent toltsuk be a buffert, itt csak egz tombkent
    // 9. upload to buffer
    // should give array size
    // we have static data, so GL_STATIC_DRAW should be used
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex), data, GL_STATIC_DRAW);

    // 12. how we want to bind buffers for position
    // parameters:
    // which slot should be used
    // how many value has a vertex
    // type of these
    //
    // size of step
    // void*, in what data structure we store
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    // 13. enable 0th attribute slot
    glEnableVertexAttribArray(0);
    // 14. how we bind for colors
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, color));
    glEnableVertexAttribArray(1);

    // 15. (+)
    // what are shaders?
    // small c-like programs
    // two types:
    // vertex shader: tells where a vertex goes on the screen
    // fragment shader: positions known, gpu knows which pixel should be drawn, then comes this shader, and tells color
    // of pixel
    // vertex shader -> out?
    // fragment shader -> in
    // a lot of math types
    // vec2, 3, 4, mat2, 3, 4
    // 4 basic operations (if it makes sense)
    // vector multiplication
    // dot, cross, length etc.
    // in vertex shader: in

}

void onDisplay() {
    glutSwapBuffers(); // 2. ahelyett hogy a kepernypre, inkabb extra memoria a kartyan, a kepernyon elozo kocka smooth
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

void onIdle() {
}
