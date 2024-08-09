// Jogo da velha criado por Lucas Elias de Andrade Cruvinel
// Email : LucrilhasBR@hotmail.com ou LucrilhasBR@gmail.com
// Obs : Jogo feito para windowns, nao ira funcionar como deveria em outras sistemas

// A seguir estão as bibliotecas uadas:
#include <stdlib.h>
#include <stdio.h>
#include <locale.h>
#include <windows.h>
#include<time.h>

//Este arquivo tem como principal função redirecionar o jogador caso ele queira ser X ou O e decidir sua dificuldade

void verificador_xvspc(char velha[3][3], char nomeplayer[10], int  round){ // Função que acontece após cada rodada para checar se o jogador ganhou ou perdeu se ele estiver de X

              if(velha[0][0]=='X' && velha[0][1]=='X' && velha[0][2] == 'X'){vitoria(velha, nomeplayer);}
    else if(velha[1][0]=='X' && velha[1][1]=='X' && velha[1][2] == 'X'){vitoria(velha, nomeplayer);}
    else if(velha[2][0]=='X' && velha[2][1]=='X' && velha[2][2] == 'X'){vitoria(velha, nomeplayer);}

    else if(velha[0][0]=='X' && velha[1][0]=='X' && velha[2][0] == 'X'){vitoria(velha, nomeplayer);}
    else if(velha[0][1]=='X' && velha[1][1]=='X' && velha[2][1] == 'X'){vitoria(velha, nomeplayer);}
    else if(velha[0][2]=='X' && velha[1][2]=='X' && velha[2][2] == 'X'){vitoria(velha, nomeplayer);}

    else if(velha[0][0]=='X' && velha[1][1]=='X' && velha[2][2] == 'X'){vitoria(velha, nomeplayer);}
    else if(velha[2][0]=='X' && velha[1][1]=='X' && velha[0][2] == 'X'){vitoria(velha, nomeplayer);}

    else if(velha[0][0]=='O' && velha[0][1]=='O' && velha[0][2] == 'O'){derrota(velha, nomeplayer);}
    else if(velha[1][0]=='O' && velha[1][1]=='O' && velha[1][2] == 'O'){derrota(velha, nomeplayer);}
    else if(velha[2][0]=='O' && velha[2][1]=='O' && velha[2][2] == 'O'){derrota(velha, nomeplayer);}

    else if(velha[0][0]=='O' && velha[1][0]=='O' && velha[2][0] == 'O'){derrota(velha, nomeplayer);}
    else if(velha[0][1]=='O' && velha[1][1]=='O' && velha[2][1] == 'O'){derrota(velha, nomeplayer);}
    else if(velha[0][2]=='O' && velha[1][2]=='O' && velha[2][2] == 'O'){derrota(velha, nomeplayer);}

    else if(velha[0][0]=='O' && velha[1][1]=='O' && velha[2][2] == 'O'){derrota(velha, nomeplayer);}
    else if(velha[2][0]=='O' && velha[1][1]=='O' && velha[0][2] == 'O'){derrota(velha, nomeplayer);}

    else if(round == 9){empate_pc(velha, nomeplayer);}
}

void verificador_ovspc(char velha[3][3], char nomeplayer[10], int  round){ // Função que acontece após cada rodada para checar se o jogador ganhou ou perdeu se ele estiver de O

              if(velha[0][0]=='X' && velha[0][1]=='X' && velha[0][2] == 'X'){derrota(velha, nomeplayer);}
    else if(velha[1][0]=='X' && velha[1][1]=='X' && velha[1][2] == 'X'){derrota(velha, nomeplayer);}
    else if(velha[2][0]=='X' && velha[2][1]=='X' && velha[2][2] == 'X'){derrota(velha, nomeplayer);}

    else if(velha[0][0]=='X' && velha[1][0]=='X' && velha[2][0] == 'X'){derrota(velha, nomeplayer);}
    else if(velha[0][1]=='X' && velha[1][1]=='X' && velha[2][1] == 'X'){derrota(velha, nomeplayer);}
    else if(velha[0][2]=='X' && velha[1][2]=='X' && velha[2][2] == 'X'){derrota(velha, nomeplayer);}

    else if(velha[0][0]=='X' && velha[1][1]=='X' && velha[2][2] == 'X'){derrota(velha, nomeplayer);}
    else if(velha[2][0]=='X' && velha[1][1]=='X' && velha[0][2] == 'X'){derrota(velha, nomeplayer);}

    else if(velha[0][0]=='O' && velha[0][1]=='O' && velha[0][2] == 'O'){vitoria(velha, nomeplayer);}
    else if(velha[1][0]=='O' && velha[1][1]=='O' && velha[1][2] == 'O'){vitoria(velha, nomeplayer);}
    else if(velha[2][0]=='O' && velha[2][1]=='O' && velha[2][2] == 'O'){vitoria(velha, nomeplayer);}

    else if(velha[0][0]=='O' && velha[1][0]=='O' && velha[2][0] == 'O'){vitoria(velha, nomeplayer);}
    else if(velha[0][1]=='O' && velha[1][1]=='O' && velha[2][1] == 'O'){vitoria(velha, nomeplayer);}
    else if(velha[0][2]=='O' && velha[1][2]=='O' && velha[2][2] == 'O'){vitoria(velha, nomeplayer);}

    else if(velha[0][0]=='O' && velha[1][1]=='O' && velha[2][2] == 'O'){vitoria(velha, nomeplayer);}
    else if(velha[2][0]=='O' && velha[1][1]=='O' && velha[0][2] == 'O'){vitoria(velha, nomeplayer);}

    else if(round == 9){empate_pc(velha, nomeplayer);}
}


void vs_pc(){ // Função que começa o jogo contra a máquina
    system("cls"); system("color 07"); setlocale(LC_ALL, "Portuguese");

    int i, j, k, signal; //Declara as variaveis
    char sinal;
    char nomeplayer[10];
    char velha[3][3], amostrador[3][3];

    for(i=0, k=49;i<3;i++){for(j=0;j<3;j++){velha[i][j] = NULL;amostrador[i][j] = k;k++;}}//Colocando NULL na matriz na velha, e Botando numeros de 1 a 9 no amostrador

 	printf("\n > Nome do player:  "); //Pegando nome dos players
 	scanf("%s", &nomeplayer);
 	printf("\n > Desejar ser X ou O?  ");
 	scanf("%s", &sinal);
 	signal = sinal;

 	if(signal == 'x' || signal == 'X'){x_vs_pc(velha, nomeplayer, amostrador);} // Cao o jogador queira ser X é redirecionado para o menu de X
 	if(signal == 'o' || signal == 'O'){o_vs_pc(velha, nomeplayer, amostrador);} // Cao o jogador queira ser O é redirecionado para o menu de O

}

void x_vs_pc(char velha[3][3], char nomeplayer[10], char amostrador[3][3]){ // Função menu de X
    setlocale(LC_ALL, "Portuguese"); system("color 07");

    int dificescolhida, i=1; // Declara variaveis

    while(i){   //menu para que o jogador escolha a dificuldade
        system("cls");
        printf("\n > Qual será a dificuldade?");
        printf("\n   > Pressione 1 para fácil.");
        printf("\n   > Pressione 2 para médio.");
        printf("\n   > Pressione 3 para difícil.\n");

        dificescolhida = getch(); // pega a escolha do jogador

        if(dificescolhida == 49)             {x_vs_pc_rounds_facil(velha, amostrador, nomeplayer, 1);} // Função que vai para o jogo facil como X
        else if(dificescolhida == 50)   {x_vs_pc_rounds_medio(velha, amostrador, nomeplayer, 1);}// Função que vai para o jogo medio como X
        else if(dificescolhida == 51)   {x_vs_pc_rounds_dificil(velha, amostrador, nomeplayer, 1);}// Função que vai para o jogo dificil como X
        else{printf("\n\n > Por favor digite  algo válido!\n\n");}

        system("pause");

    }
}

void o_vs_pc(char velha[3][3], char nomeplayer[10], char amostrador[3][3]){
    setlocale(LC_ALL, "Portuguese"); system("color 07");
    int dificescolhida, i=1;

    while(i){ //menu para que o jogador escolha a dificuldade
        system("cls");
        printf("\n > Qual será a dificuldade?");
        printf("\n   > Pressione 1 para fácil.");
        printf("\n   > Pressione 2 para médio.");
        printf("\n   > Pressione 3 para difícil.\n");

        dificescolhida = getch();// pega a escolha do jogador

        if(dificescolhida == 49){o_vs_pc_rounds_facil(velha, amostrador, nomeplayer, 1);}// Função que vai para o jogo facil como O
        else if(dificescolhida == 50){o_vs_pc_rounds_medio(velha, amostrador, nomeplayer, 1);}// Função que vai para o jogo medio como O
        else if(dificescolhida == 51){o_vs_pc_rounds_dificil(velha, amostrador, nomeplayer, 1);}// Função que vai para o jogo dificil como O
        else{printf("\n\n > Por favor digite  algo válido!\n\n");}

        system("pause");

    }
}

