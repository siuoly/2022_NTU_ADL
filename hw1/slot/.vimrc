



nnoremap <space>w :execute 'SlimeSend1 '.. expand("<cword>")<cr>
nnoremap <space>W :execute 'SlimeSend1 '.. expand("<cWORD>")<cr>
nnoremap <space>p :execute 'SlimeSend1 '.. 'print('.. expand("<cword>") ..')'<cr>

" variable explorer
" set autoread
" command! -nargs=1 VariableInspect  SlimeSend0 "%store ".<f-args>." >>log.txt"|checktime

" run file
" let @w="w@e"
" let @e='Slie€kbmeSend1 %run dataset'.."\<cr>l"
