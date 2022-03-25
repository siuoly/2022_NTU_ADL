let SessionLoad = 1
if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
inoremap <silent> <Plug>(-fzf-complete-finish) l
inoremap <silent> <expr> <BS> coc#_insert_key('request', 'iPGJzPg==0')
inoremap <silent> <Plug>CocRefresh =coc#_complete()
inoremap <silent> <Plug>(fzf-maps-i) :call fzf#vim#maps('i', 0)
inoremap <expr> <Plug>(fzf-complete-buffer-line) fzf#vim#complete#buffer_line()
inoremap <expr> <Plug>(fzf-complete-line) fzf#vim#complete#line()
inoremap <expr> <Plug>(fzf-complete-file-ag) fzf#vim#complete#path('ag -l -g ""')
inoremap <expr> <Plug>(fzf-complete-file) fzf#vim#complete#path("find . -path '*/\.*' -prune -o -type f -print -o -type l -print | sed 's:^..::'")
inoremap <expr> <Plug>(fzf-complete-path) fzf#vim#complete#path("find . -path '*/\.*' -prune -o -print | sed '1d;s:^..::'")
inoremap <expr> <Plug>(fzf-complete-word) fzf#vim#complete#word()
inoremap <silent> <C-J> =UltiSnips#ListSnippets()
inoremap <silent> <C-L> =UltiSnips#ExpandSnippet()
imap <C-G>S <Plug>ISurround
imap <C-G>s <Plug>Isurround
imap <C-S> <Plug>Isurround
cnoremap <expr> <C-R><C-O><C-P> traces#check_b() ? "\\=traces#get_pfile()\" : "\\\"
cnoremap <expr> <C-R><C-O><C-F> traces#check_b() ? "\\=traces#get_cfile()\" : "\\\"
cnoremap <expr> <C-R><C-O><C-A> traces#check_b() ? "\\=traces#get_cWORD()\" : "\\\"
cnoremap <expr> <C-R><C-O><C-W> traces#check_b() ? "\\=traces#get_cword()\" : "\\\"
cnoremap <expr> <C-R><C-R><C-P> traces#check_b() ? "\\=traces#get_pfile()\" : "\\\"
cnoremap <expr> <C-R><C-R><C-F> traces#check_b() ? "\\=traces#get_cfile()\" : "\\\"
cnoremap <expr> <C-R><C-R><C-A> traces#check_b() ? "\\=traces#get_cWORD()\" : "\\\"
cnoremap <expr> <C-R><C-R><C-W> traces#check_b() ? "\\=traces#get_cword()\" : "\\\"
cnoremap <expr> <C-R><C-P> traces#check_b() ? traces#get_pfile() : "\\"
cnoremap <expr> <C-R><C-F> traces#check_b() ? traces#get_cfile() : "\\"
cnoremap <expr> <C-R><C-A> traces#check_b() ? traces#get_cWORD() : "\\"
cnoremap <expr> <C-R><C-W> traces#check_b() ? traces#get_cword() : "\\"
inoremap <expr> <S-Tab> pumvisible() ? "\" : "\"
cnoremap <M-l> <Right>
cnoremap <M-h> <Left>
cnoremap <M-d> <C-Right>
cnoremap <M-f> <C-Right>
cnoremap <M-b> <C-Left>
cnoremap <C-P> <Up>
cnoremap <C-N> <Down>
cnoremap <C-E> <End>
cnoremap <C-A> <Home>
inoremap <M-K> <Home>
inoremap <M-J> <End>
inoremap <M-L> <C-Right>
inoremap <M-H> <C-Left>
inoremap <M-k> <Up>
inoremap <M-j> <Down>
inoremap <M-l> <Right>
inoremap <M-h> <Left>
inoremap <C-U> u
nnoremap  :AsyncStop
tnoremap <silent>  :tabp
snoremap <silent>  "_c
nnoremap <silent>  :tabp
xnoremap 	 >gv
snoremap <silent> <NL> :call UltiSnips#ListSnippets()
nnoremap <silent> <NL> :silent n | ar
nnoremap <silent>  :silent N | ar
tnoremap <silent>  :tabn
xnoremap <silent>  :call UltiSnips#SaveLastVisualSelection()gvs
snoremap <silent>  :call UltiSnips#ExpandSnippet()
nnoremap <silent>  :tabn
xnoremap  :
nnoremap <expr>  (&ft=="qf" || &bt=="nofile") ? "":":"
nmap  <Plug>(RepeatRedo)
snoremap  "_c
nnoremap  T
nnoremap  c :SlimeSend1 
nnoremap <expr>  r ":SlimeSend1 run " ..expand('%') .. ""
nnoremap    :IPythonCellExecuteCell
nnoremap  b o3i#j
nnoremap  k :IPythonCellPrevCell
nnoremap  j :IPythonCellNextCell
xnoremap <expr>   UltiSnips#CanExpandSnippet()? "\=UltiSnips#ExpandSnippetOrJump()\":" "
nmap "W ysiw"
nnoremap <silent> '[ :call signature#mark#Goto("prev", "line", "alpha")
nnoremap <silent> '] :call signature#mark#Goto("next", "line", "alpha")
nnoremap * *``
nmap . <Plug>(RepeatDot)
nnoremap <expr> : (&bt=="") ? "," : ":"
xnoremap < <gv
xnoremap > >gv
onoremap B iB
cnoremap ì <Right>
cnoremap è <Left>
cnoremap ä <C-Right>
cnoremap æ <C-Right>
cnoremap â <C-Left>
inoremap Ë <Home>
inoremap Ê <End>
inoremap Ì <C-Right>
inoremap È <C-Left>
inoremap ë <Up>
inoremap ê <Down>
inoremap ì <Right>
inoremap è <Left>
nnoremap H ^
xnoremap H ^
onoremap H ^
nnoremap <expr> J "m'" . v:count . "J`'"
nnoremap L g_
xnoremap L g_
onoremap L g_
nnoremap N Nzz
nnoremap Q q
xmap Q gq
omap Q gq
xmap S <Plug>VSurround
nmap U <Plug>(RepeatUndoLine)
nnoremap Y y$
nnoremap ZZ :wall | qa
nnoremap <silent> [= :call signature#marker#Goto("prev", "any",  v:count)
nnoremap <silent> [- :call signature#marker#Goto("prev", "same", v:count)
nnoremap <silent> [` :call signature#mark#Goto("prev", "spot", "pos")
nnoremap <silent> [' :call signature#mark#Goto("prev", "line", "pos")
nnoremap \a :tabe | CocConfig
nnoremap \u <Cmd>UltiSnipsEdit
nnoremap \G :call Togglefile( $VIMFILES .. '/tasks.ini')
nnoremap \g :call Togglefile('.tasks')
nnoremap \c :tabe |CocConfig
nnoremap \z <Cmd>ZoomToggle
nnoremap \e :NERDTreeToggle
nnoremap \p :call Togglefile($VIMFILES .. '/after/plug.vim')
nnoremap \m :tabedit $VIMFILES/after/commonMap.vim
nnoremap \v :call Togglefile($VIMFILES .. '/vimrc')
nnoremap \t :call Togglefile($VIMFILES .. "/templates/template." .&ft)
nnoremap \V :exec "tabe $VIMFILES/after/ftplugin/" .&ft .".vim"
nnoremap \q q
nnoremap <silent> ]= :call signature#marker#Goto("next", "any",  v:count)
nnoremap <silent> ]- :call signature#marker#Goto("next", "same", v:count)
nnoremap <silent> ]` :call signature#mark#Goto("next", "spot", "pos")
nnoremap <silent> ]' :call signature#mark#Goto("next", "line", "pos")
nnoremap <silent> `[ :call signature#mark#Goto("prev", "spot", "alpha")
nnoremap <silent> `] :call signature#mark#Goto("next", "spot", "alpha")
onoremap <silent> al :normal! $v0
xnoremap al :normal! $v0
onoremap b i(
nmap cS <Plug>CSurround
nmap cs <Plug>Csurround
nnoremap <silent> dm :call signature#utils#Remove(v:count)
nmap ds <Plug>Dsurround
xnoremap f zf
xmap gx <Plug>NetrwBrowseXVis
xmap gS <Plug>VgSurround
xnoremap gsd :AsyncRun -silent gtts-cli - | mpg123.exe -q - 
nnoremap gsd :AsyncRun -mode=term -pos=hide gtts-cli <cword> | mpg123.exe -q - 
xnoremap gss :AsyncRun -cwd=$VIMFILES/script ./translateZH_TW.py -
nnoremap gss :AsyncRun -cwd=$VIMFILES/script ./translateZH_TW.py <cword>
nmap gcc <Plug>CommentaryLine
omap gc <Plug>Commentary
nmap gc <Plug>Commentary
xmap gc <Plug>Commentary
xnoremap gse :AsyncRun cat | xargs -0 -I {} translate {}
nnoremap gse :AsyncRun translate <cword>
nnoremap gx :call system("xdg-open ".expand('<cWORD>'))
nnoremap g; 2g;
xnoremap ii `[v`]h
xnoremap i/ ?\/\*o/\*\//s+1
onoremap <silent> id :normal! GVgg
xnoremap <silent> id :normal! G$Vgg0
onoremap <silent> il :normal! g_v^
xnoremap il :normal! g_v^
nnoremap j gj
nnoremap k gk
nnoremap <silent> m? :call signature#marker#List(v:count, 0)
nnoremap <silent> m/ :call signature#mark#List(0, 0)
nnoremap <silent> m<BS> :call signature#marker#Purge()
nnoremap <silent> m  :call signature#mark#Purge("all")
nnoremap <silent> m- :call signature#mark#Purge("line")
nnoremap <silent> m. :call signature#mark#ToggleAtLine()
nnoremap <silent> m, :call signature#mark#Toggle("next")
nnoremap <silent> m :call signature#utils#Input()
nnoremap n nzz
onoremap p ip
nnoremap p gp
nmap u <Plug>(RepeatUndo)
nmap ySS <Plug>YSsurround
nmap ySs <Plug>YSsurround
nmap yss <Plug>Yssurround
nmap yS <Plug>YSurround
nmap ys <Plug>Ysurround
nnoremap <silent> <Plug>(-fzf-complete-finish) a
nnoremap <Plug>(-fzf-:) :
nnoremap <Plug>(-fzf-/) /
nnoremap <Plug>(-fzf-vim-do) :execute g:__fzf_command
nmap <C-R> <Plug>(RepeatRedo)
nnoremap <silent> <Plug>(RepeatRedo) :call repeat#wrap("\<C-R>",v:count)
nnoremap <silent> <Plug>(RepeatUndoLine) :call repeat#wrap('U',v:count)
nnoremap <silent> <Plug>(RepeatUndo) :call repeat#wrap('u',v:count)
nnoremap <silent> <Plug>(RepeatDot) :if !repeat#run(v:count)|echoerr repeat#errmsg()|endif
nmap <silent> <Plug>CommentaryUndo :echoerr "Change your <Plug>CommentaryUndo map to <Plug>Commentary<Plug>Commentary"
onoremap <silent> <Plug>(coc-classobj-a) :call CocAction('selectSymbolRange', v:false, '', ['Interface', 'Struct', 'Class'])
onoremap <silent> <Plug>(coc-classobj-i) :call CocAction('selectSymbolRange', v:true, '', ['Interface', 'Struct', 'Class'])
vnoremap <silent> <Plug>(coc-classobj-a) :call CocAction('selectSymbolRange', v:false, visualmode(), ['Interface', 'Struct', 'Class'])
vnoremap <silent> <Plug>(coc-classobj-i) :call CocAction('selectSymbolRange', v:true, visualmode(), ['Interface', 'Struct', 'Class'])
onoremap <silent> <Plug>(coc-funcobj-a) :call CocAction('selectSymbolRange', v:false, '', ['Method', 'Function'])
onoremap <silent> <Plug>(coc-funcobj-i) :call CocAction('selectSymbolRange', v:true, '', ['Method', 'Function'])
vnoremap <silent> <Plug>(coc-funcobj-a) :call CocAction('selectSymbolRange', v:false, visualmode(), ['Method', 'Function'])
vnoremap <silent> <Plug>(coc-funcobj-i) :call CocAction('selectSymbolRange', v:true, visualmode(), ['Method', 'Function'])
nnoremap <silent> <Plug>(coc-cursors-position) :call CocAction('cursorsSelect', bufnr('%'), 'position', 'n')
nnoremap <silent> <Plug>(coc-cursors-word) :call CocAction('cursorsSelect', bufnr('%'), 'word', 'n')
vnoremap <silent> <Plug>(coc-cursors-range) :call CocAction('cursorsSelect', bufnr('%'), 'range', visualmode())
nnoremap <silent> <Plug>(coc-refactor) :call       CocActionAsync('refactor')
nnoremap <silent> <Plug>(coc-command-repeat) :call       CocAction('repeatCommand')
nnoremap <silent> <Plug>(coc-float-jump) :call       coc#float#jump()
nnoremap <silent> <Plug>(coc-float-hide) :call       coc#float#close_all()
nnoremap <silent> <Plug>(coc-fix-current) :call       CocActionAsync('doQuickfix')
nnoremap <silent> <Plug>(coc-openlink) :call       CocActionAsync('openLink')
nnoremap <silent> <Plug>(coc-references-used) :call       CocActionAsync('jumpUsed')
nnoremap <silent> <Plug>(coc-references) :call       CocActionAsync('jumpReferences')
nnoremap <silent> <Plug>(coc-type-definition) :call       CocActionAsync('jumpTypeDefinition')
nnoremap <silent> <Plug>(coc-implementation) :call       CocActionAsync('jumpImplementation')
nnoremap <silent> <Plug>(coc-declaration) :call       CocActionAsync('jumpDeclaration')
nnoremap <silent> <Plug>(coc-definition) :call       CocActionAsync('jumpDefinition')
nnoremap <silent> <Plug>(coc-diagnostic-prev-error) :call       CocActionAsync('diagnosticPrevious', 'error')
nnoremap <silent> <Plug>(coc-diagnostic-next-error) :call       CocActionAsync('diagnosticNext',     'error')
nnoremap <silent> <Plug>(coc-diagnostic-prev) :call       CocActionAsync('diagnosticPrevious')
nnoremap <silent> <Plug>(coc-diagnostic-next) :call       CocActionAsync('diagnosticNext')
nnoremap <silent> <Plug>(coc-diagnostic-info) :call       CocActionAsync('diagnosticInfo')
nnoremap <silent> <Plug>(coc-format) :call       CocActionAsync('format')
nnoremap <silent> <Plug>(coc-rename) :call       CocActionAsync('rename')
nnoremap <Plug>(coc-codeaction-cursor) :call       CocActionAsync('codeAction',         'cursor')
nnoremap <Plug>(coc-codeaction-line) :call       CocActionAsync('codeAction',         'line')
nnoremap <Plug>(coc-codeaction) :call       CocActionAsync('codeAction',         '')
vnoremap <silent> <Plug>(coc-codeaction-selected) :call       CocActionAsync('codeAction',         visualmode())
vnoremap <silent> <Plug>(coc-format-selected) :call       CocActionAsync('formatSelected',     visualmode())
nnoremap <Plug>(coc-codelens-action) :call       CocActionAsync('codeLensAction')
nnoremap <Plug>(coc-range-select) :call       CocActionAsync('rangeSelect',     '', v:true)
vnoremap <silent> <Plug>(coc-range-select-backward) :call       CocActionAsync('rangeSelect',     visualmode(), v:false)
vnoremap <silent> <Plug>(coc-range-select) :call       CocActionAsync('rangeSelect',     visualmode(), v:true)
noremap <SNR>60_Operator :call slime#store_curpos():set opfunc=slime#send_opg@
xnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
tnoremap <silent> <M-=> :call TerminalToggle()
nnoremap <silent> <M-=> :call TerminalToggle()
tnoremap <silent> <C-L> :tabn
tnoremap <silent> <C-H> :tabp
tnoremap <F2> N  
tnoremap <M-p> "0
tnoremap <M--> "0
tnoremap <M-N> p
tnoremap <M-k> k
tnoremap <M-j> j
tnoremap <M-l> l
tnoremap <M-h> h
tnoremap <silent> <Plug>(fzf-normal) 
tnoremap <silent> <Plug>(fzf-insert) i
nnoremap <silent> <Plug>(fzf-normal) <Nop>
nnoremap <silent> <Plug>(fzf-insert) i
onoremap <silent> <Plug>(fzf-maps-o) :call fzf#vim#maps('o', 0)
xnoremap <silent> <Plug>(fzf-maps-x) :call fzf#vim#maps('x', 0)
nnoremap <silent> <Plug>(fzf-maps-n) :call fzf#vim#maps('n', 0)
snoremap <C-R> "_c
snoremap <silent> <C-H> "_c
snoremap <silent> <Del> "_c
snoremap <silent> <BS> "_c
snoremap <silent> <C-J> :call UltiSnips#ListSnippets()
xnoremap <silent> <C-L> :call UltiSnips#SaveLastVisualSelection()gvs
snoremap <silent> <C-L> :call UltiSnips#ExpandSnippet()
nnoremap <silent> <Plug>SurroundRepeat .
tnoremap <F3> <Cmd>ZoomToggle
nnoremap <expr> <F3> &bt=="" ? "<Cmd>ZoomToggle":"<Cmd>ZoomToggle"
nnoremap <expr> <F5> &bt=="" ? "<Cmd>call JumpQuickfixWin() | ZoomToggle" : ":ZoomToggle"
nnoremap <C-C> :AsyncStop
tnoremap <M-w> <Cmd>wincmd p
nnoremap <expr> <F4> &bt=="" ? ":call JumpTerminalWin():ZoomToggle" : "<Cmd>ZoomTogglei<Cmd>wincmd p"
map <M-RightMouse> u
xnoremap <RightMouse> y
nnoremap <RightMouse> p 
xnoremap <S-Tab> <gv
tnoremap <C-Down> <Cmd> res -1
tnoremap <C-Up> <Cmd> res +1
tnoremap <M-.> <Cmd>vert res +1
tnoremap <M-,> <Cmd>vert res -1
tnoremap <C-Right> <Cmd>vert res +1
tnoremap <C-Left> <Cmd>vert res -1
nnoremap <M-,> <<Cmd>echo "using c-arrow"
nnoremap <M-.> ><Cmd>echo "using c-arrow"
nnoremap <C-Down> -
nnoremap <C-Up> +
nnoremap <C-Right> >
nnoremap <C-Left> <
nnoremap <silent> <M-l> :wincmd l
nnoremap <silent> <M-k> :wincmd k
nnoremap <silent> <M-j> :wincmd j
nnoremap <silent> <M-h> :wincmd h
xnoremap <M-K> :m '<-2gv
xnoremap <M-J> :m '>+1gv
nnoremap <silent> <M-J> :m .+1
nnoremap <silent> <M-K> :m .-2
nnoremap <silent> <C-K> :silent N | ar
nnoremap <silent> <C-J> :silent n | ar
nnoremap <M-b> @b
nnoremap <M-n> @n
nnoremap <M-p> :bp 
nnoremap <C-W><C-T> T
nnoremap <silent> <C-L> :tabn
nnoremap <silent> <C-H> :tabp
nnoremap <M-4> :w:exe getline("4")[2:]
nnoremap <M-3> :w:exe getline("3")[2:]
nnoremap <M-2> :w:exe getline("2")[2:]
nnoremap <M-1> :w:exe getline("1")[2:]
nnoremap <silent> <F6> :Tlist
nnoremap <expr> <F2> &bt!="terminal" ? ":nohlsearch" : "i"
nnoremap <Down> 
nnoremap <Up> 
nnoremap <BS> 
tnoremap <M-q> 
nnoremap <M-q> :q
nnoremap <M-w> :wall
cnoremap  <Home>
cnoremap  <End>
imap S <Plug>ISurround
imap s <Plug>Isurround
inoremap <silent> <NL> =UltiSnips#ListSnippets()
inoremap <silent>  =UltiSnips#ExpandSnippet()
cnoremap  <Down>
cnoremap  <Up>
cnoremap <expr>  traces#check_b() ? "\\=traces#get_pfile()\" : "\\\"
cnoremap <expr>  traces#check_b() ? "\\=traces#get_cfile()\" : "\\\"
cnoremap <expr>  traces#check_b() ? "\\=traces#get_cWORD()\" : "\\\"
cnoremap <expr>  traces#check_b() ? "\\=traces#get_cword()\" : "\\\"
cnoremap <expr>  traces#check_b() ? "\\=traces#get_pfile()\" : "\\\"
cnoremap <expr>  traces#check_b() ? "\\=traces#get_cfile()\" : "\\\"
cnoremap <expr>  traces#check_b() ? "\\=traces#get_cWORD()\" : "\\\"
cnoremap <expr>  traces#check_b() ? "\\=traces#get_cword()\" : "\\\"
cnoremap <expr>  traces#check_b() ? traces#get_pfile() : "\\"
cnoremap <expr>  traces#check_b() ? traces#get_cfile() : "\\"
cnoremap <expr>  traces#check_b() ? traces#get_cWORD() : "\\"
cnoremap <expr>  traces#check_b() ? traces#get_cword() : "\\"
imap  <Plug>Isurround
inoremap  u
inoremap <expr>   UltiSnips#CanExpandSnippet()? "\=UltiSnips#ExpandSnippetOrJump()\":" "
inoremap <silent> <expr> " coc#_insert_key('request', 'iIg==0')
inoremap <silent> <expr> ' coc#_insert_key('request', 'iJw==0')
inoremap <silent> <expr> ( coc#_insert_key('request', 'iKA==0')
inoremap <silent> <expr> ) coc#_insert_key('request', 'iKQ==0')
inoremap , ,u
inoremap . .u
inoremap ; ;u
inoremap <silent> <expr> < coc#_insert_key('request', 'iPA==0')
inoremap <silent> <expr> > coc#_insert_key('request', 'iPg==0')
tnoremap <silent> ½ :call TerminalToggle()
nnoremap <silent> ½ :call TerminalToggle()
tnoremap ­ "0
tnoremap ® <Cmd>vert res +1
tnoremap ¬ <Cmd>vert res -1
nnoremap ¬ <<Cmd>echo "using c-arrow"
nnoremap ® ><Cmd>echo "using c-arrow"
nnoremap ´ :w:exe getline("4")[2:]
nnoremap ³ :w:exe getline("3")[2:]
nnoremap ² :w:exe getline("2")[2:]
nnoremap ± :w:exe getline("1")[2:]
tnoremap ð "0
tnoremap Î p
tnoremap ë k
tnoremap ê j
tnoremap ì l
tnoremap è h
tnoremap ÷ <Cmd>wincmd p
nnoremap <silent> ì :wincmd l
nnoremap <silent> ë :wincmd k
nnoremap <silent> ê :wincmd j
nnoremap <silent> è :wincmd h
xnoremap Ë :m '<-2gv
xnoremap Ê :m '>+1gv
nnoremap <silent> Ê :m .+1
nnoremap <silent> Ë :m .-2
nnoremap â @b
nnoremap î @n
nnoremap ð :bp 
tnoremap ñ 
nnoremap ñ :q
nnoremap ÷ :wall
inoremap <silent> <expr> [ coc#_insert_key('request', 'iWw==0')
inoremap <silent> <expr> ] coc#_insert_key('request', 'iXQ==0')
inoremap <silent> <expr> ` coc#_insert_key('request', 'iYA==0')
cnoremap qw silent! wall |bufdo qa!
cnoremap qq q!
inoremap <silent> <expr> { coc#_insert_key('request', 'iew==0')
inoremap <silent> <expr> } coc#_insert_key('request', 'ifQ==0')
iabbr nmap nnoremap <bufuer>
iabbr #i import
cabbr w!! w !sudo tee %
cabbr term vert term
cabbr th tab h
cabbr h vert bo h
cabbr ww silent! wall
let &cpo=s:cpo_save
unlet s:cpo_save
set ambiwidth=double
set autoindent
set backspace=indent,eol,start
set belloff=all
set clipboard=unnamedplus
set complete=.,w,b,u
set cpoptions=aABceFsy
set display=truncate
set expandtab
set exrc
set fileencodings=ucs-bom,utf-8,default,latin1
set helplang=cn
set hidden
set history=200
set hlsearch
set ignorecase
set incsearch
set langnoremap
set nolangremap
set laststatus=2
set mouse=a
set nrformats=bin,hex
set operatorfunc=<SNR>89_go
set path=.,/usr/include,,,~/.local/lib/python3.10/site-packages/_pdbpp_path_hack,/usr/lib/python3.10,/usr/lib/python3.10/lib-dynload,~/.local/lib/python3.10/site-packages,/usr/lib/python3.10/site-packages
set ruler
set runtimepath=~/.vim,~/.vim/plugged/vim-misc,~/.vim/plugged/traces.vim,~/.vim/plugged/indentLine,~/.vim/plugged/vim-commentary,~/.vim/plugged/vim-context-commentstring,~/.vim/plugged/vim-surround,~/.vim/plugged/vim-repeat,~/.vim/plugged/better-indent-support-for-php-with-html,~/.vim/plugged/ultisnips,~/.vim/plugged/fzf.vim,~/.vim/plugged/fzf,~/.vim/plugged/folddigest.vim,~/.vim/plugged/vim-closetag,~/.vim/plugged/vim-terminal-help,~/.vim/plugged/vim-signature,~/.vim/plugged/coc.nvim,~/.vim/plugged/vimcdoc,~/.vim/plugged/vim-colorschemes,~/.vim/plugged/jupyter-vim,~/.vim/plugged/jupytext.vim,~/.vim/plugged/vim-ipython-cell,~/.vim/plugged/vim-slime,~/.vim/plugged/md-img-paste.vim,~/.vim/plugged/markdown-preview.nvim,/share/vim/vimfiles,/share/vim/vim82,/share/vim/vimfiles/after,~/.vim/plugged/indentLine/after,~/.vim/plugged/ultisnips/after,~/.vim/plugged/vim-signature/after,~/.vim/after
set scrolloff=9
set secure
set shiftwidth=2
set shortmess=filnxtToOSIc
set showcmd
set smartcase
set softtabstop=2
set statusline=%-15f%h%m%r%w%=[%{&fileencoding}]%-6y--%p%%[%l/%L]
set tabstop=2
set ttimeout
set ttimeoutlen=85
set updatetime=300
set wildignore=*.o,*.class,*.pyc
set wildmenu
set window=43
set winminheight=0
set winminwidth=0
set nowritebackup
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd /mnt/d/grad/ADL/hw1-2/ADL21-HW1/slot
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +136 slot.py
badd +173 ~/.vim/after/commonMap.vim
badd +47 ~/.vim/after/ftplugin/python.vim
badd +3 ~/.vim/after/ftplugin/vim.vim
badd +0 ~/.vim/after/vim.vim
badd +1 ~/.vim/after/python.vim
badd +2 ~/.vim/UltiSnips/python.snippets
badd +0 mk
argglobal
%argdel
$argadd slot.py
set stal=2
tabnew +setlocal\ bufhidden=wipe
tabrewind
edit slot.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 85 + 80) / 161)
exe 'vert 2resize ' . ((&columns * 75 + 80) / 161)
argglobal
balt ~/.vim/UltiSnips/python.snippets
nnoremap <buffer>  :exe "pyx "..getline(".")
nnoremap <buffer> <silent> \U :JupyterUpdateShell
let s:cpo_save=&cpo
set cpo&vim
vmap <buffer> <silent> \e <Plug>JupyterRunVisual
nmap <buffer> <silent> \e <Plug>JupyterRunTextObj
nnoremap <buffer> <silent> \E :JupyterSendRange
nnoremap <buffer> <silent> \X :JupyterSendCell
nnoremap <buffer> <silent> \d :JupyterCd %:p:h
nnoremap <buffer> <silent> \R :JupyterRunFile
nnoremap <buffer> <silent> \b :PythonSetBreak
nnoremap <buffer> <silent> \I :PythonImportThisFile
nnoremap <buffer> <F7> :SlimeConfig H<Cmd>sb cmdT<Cmd>tabp
nnoremap <buffer> <M-s> vip:RunPy
nnoremap <buffer> <C-S> :exe "pyx "..getline(".")
map <buffer> <M-c> <Cmd>call JumpTerminalWin()
nnoremap <buffer> <M-d> :w:exec "H python " ..expand('%')
nnoremap <buffer> ó vip:RunPy
map <buffer> ã <Cmd>call JumpTerminalWin()
nnoremap <buffer> ä :w:exec "H python " ..expand('%')
let &cpo=s:cpo_save
unlet s:cpo_save
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal autoread
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal nocindent
setlocal cinkeys=0{,0},0),0],:,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=b:#,fb:-
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u
setlocal concealcursor=inc
setlocal conceallevel=2
setlocal completefunc=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
set cursorline
setlocal cursorline
setlocal cursorlineopt=both
setlocal define=^\\s*\\(def\\|class\\)
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'python'
setlocal filetype=python
endif
setlocal fixendofline
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatprg=
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=-1
setlocal include=^\\s*\\(from\\|import\\)
setlocal includeexpr=substitute(substitute(substitute(v:fname,b:grandparent_match,b:grandparent_sub,''),b:parent_match,b:parent_sub,''),b:child_match,b:child_sub,'g')
setlocal indentexpr=GetPythonIndent(v:lnum)
setlocal indentkeys=0{,0},0),0],:,0#,!^F,o,O,e,<:>,=elif,=except
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=python3\ -m\ pydoc
setlocal nolinebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal listchars=
setlocal makeencoding=
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=bin,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=python3complete#Complete
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
set relativenumber
setlocal relativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal scrolloff=-1
setlocal shiftwidth=4
setlocal noshortname
setlocal showbreak=
setlocal sidescrolloff=-1
setlocal signcolumn=auto
setlocal nosmartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal spelloptions=
setlocal statusline=
setlocal suffixesadd=.py
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'python'
setlocal syntax=python
endif
setlocal tabstop=4
setlocal tagcase=
setlocal tagfunc=
setlocal tags=
set termwinkey=<c-_>
setlocal termwinkey=<c-_>
setlocal termwinscroll=10000
setlocal termwinsize=
setlocal textwidth=0
setlocal thesaurus=
setlocal thesaurusfunc=
setlocal noundofile
setlocal undolevels=-123456
setlocal varsofttabstop=
setlocal vartabstop=
setlocal virtualedit=
setlocal wincolor=
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let &fdl = &fdl
let s:l = 106 - ((24 * winheight(0) + 20) / 41)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 106
normal! 0
wincmd w
argglobal
terminal ++curwin ++cols=75 ++rows=41 cmd.exe /k conda activate base && ipython
let s:term_buf_2 = bufnr()
balt slot.py
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=terminal
setlocal nocindent
setlocal cinkeys=0{,0},0),0],:,0#,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=s1:/*,mb:*,ex:*/,://,b:#,:%,:XCOMM,n:>,fb:-
setlocal commentstring=/*%s*/
setlocal complete=.,w,b,u
setlocal concealcursor=inc
setlocal conceallevel=2
setlocal completefunc=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal cursorlineopt=both
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != ''
setlocal filetype=
endif
setlocal fixendofline
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatprg=
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=-1
setlocal include=
setlocal includeexpr=
setlocal indentexpr=
setlocal indentkeys=0{,0},0),0],:,0#,!^F,o,O,e
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=
setlocal nolinebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal listchars=
setlocal makeencoding=
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal nomodifiable
setlocal nrformats=bin,hex
set number
setlocal nonumber
setlocal numberwidth=4
setlocal omnifunc=
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
set relativenumber
setlocal norelativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal scrolloff=-1
setlocal shiftwidth=2
setlocal noshortname
setlocal showbreak=
setlocal sidescrolloff=-1
setlocal signcolumn=auto
setlocal nosmartindent
setlocal softtabstop=2
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal spelloptions=
setlocal statusline=
setlocal suffixesadd=
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != ''
setlocal syntax=
endif
setlocal tabstop=2
setlocal tagcase=
setlocal tagfunc=
setlocal tags=
set termwinkey=<c-_>
setlocal termwinkey=<c-_>
setlocal termwinscroll=10000
setlocal termwinsize=
setlocal textwidth=0
setlocal thesaurus=
setlocal thesaurusfunc=
setlocal noundofile
setlocal undolevels=-123456
setlocal varsofttabstop=
setlocal vartabstop=
setlocal virtualedit=
setlocal wincolor=
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
let s:l = 1731 - ((40 * winheight(0) + 20) / 41)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1731
normal! 09|
wincmd w
exe 'vert 1resize ' . ((&columns * 85 + 80) / 161)
exe 'vert 2resize ' . ((&columns * 75 + 80) / 161)
tabnext
argglobal
execute 'buffer ' . s:term_buf_2
balt slot.py
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=terminal
setlocal nocindent
setlocal cinkeys=0{,0},0),0],:,0#,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=s1:/*,mb:*,ex:*/,://,b:#,:%,:XCOMM,n:>,fb:-
setlocal commentstring=/*%s*/
setlocal complete=.,w,b,u
setlocal concealcursor=inc
setlocal conceallevel=2
setlocal completefunc=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal cursorlineopt=both
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != ''
setlocal filetype=
endif
setlocal fixendofline
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatprg=
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=-1
setlocal include=
setlocal includeexpr=
setlocal indentexpr=
setlocal indentkeys=0{,0},0),0],:,0#,!^F,o,O,e
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=
setlocal nolinebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal listchars=
setlocal makeencoding=
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal nomodifiable
setlocal nrformats=bin,hex
set number
setlocal nonumber
setlocal numberwidth=4
setlocal omnifunc=
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
set relativenumber
setlocal norelativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal scrolloff=-1
setlocal shiftwidth=2
setlocal noshortname
setlocal showbreak=
setlocal sidescrolloff=-1
setlocal signcolumn=auto
setlocal nosmartindent
setlocal softtabstop=2
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal spelloptions=
setlocal statusline=
setlocal suffixesadd=
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != ''
setlocal syntax=
endif
setlocal tabstop=2
setlocal tagcase=
setlocal tagfunc=
setlocal tags=
set termwinkey=<c-_>
setlocal termwinkey=<c-_>
setlocal termwinscroll=10000
setlocal termwinsize=
setlocal textwidth=0
setlocal thesaurus=
setlocal thesaurusfunc=
setlocal noundofile
setlocal undolevels=-123456
setlocal varsofttabstop=
setlocal vartabstop=
setlocal virtualedit=
setlocal wincolor=
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
let s:l = 1 - ((0 * winheight(0) + 20) / 41)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1
normal! 0
tabnext 1
set stal=1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOSIc
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
